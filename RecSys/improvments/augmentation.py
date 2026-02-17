import torch

class Augmentation(object):
    def __init__(self, config):
        self.aug_ratio = config['aug_ratio'] if 'aug_ratio' in config else 0.2
        self.mask_token = config['mask_token'] if 'mask_token' in config else 0
        
    def _item_mask(self, item_seq, item_seq_len):
        """
        Vectorized Masking:
        Randomly mask items with 0.
        """
        ratio = self.aug_ratio
        
        # Generate random noise map [batch, seq_len]
        probs = torch.rand_like(item_seq, dtype=torch.float)
        
        # We only want to mask valid items, so we can ignore padding (0) for now
        # But efficiently: just mask where prob < ratio
        # And ensure we don't mask padding (item_seq == 0) - though masking 0 with 0 is fine.
        # Key: Don't mask EVERYTHING.
        
        mask = (probs < ratio) & (item_seq != 0)
        
        augmented_seq = item_seq.clone()
        augmented_seq[mask] = self.mask_token
        return augmented_seq

    def _item_crop(self, item_seq, item_seq_len):
        """
        Vectorized Crop (Subsequence Subsampling):
        Randomly keep items, preserve order, shift left.
        Implemented via Sorting.
        """
        batch_size, seq_len = item_seq.shape
        ratio = self.aug_ratio
        
        # 1. Determine which items to KEEP
        # Generate random scores
        keep_probs = torch.rand_like(item_seq, dtype=torch.float)
        
        # Force padding to be "removed" (moved to end) - actually padding is 0, we can handle it at end.
        # We want to keep (1-ratio) items.
        # Simpler: just threshold probability. "Drop this item with p=ratio"
        
        # keep_mask[i, j] = True if we keep item j
        keep_mask = keep_probs > ratio
        
        # Ensure we keep at least one item? Or rely on statistical likelihood?
        # Also ensure we don't 'keep' the padding 0s effectively?
        # Actually, if we treat padding as "drop", they get sorted to end anyway.
        
        # Sort key: 
        # We want kept items to maintain order: i.e., index 0 < index 1.
        # We want dropped items to move to end.
        
        # Create base indices [0, 1, 2, ... seq_len]
        indices = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        
        # Add offset to dropped items
        # If keep: score = index
        # If drop: score = index + seq_len (moves to end)
        # But padding is 0. 
        
        # Let's refine:
        # We want to essentially 'gather' the kept items.
        
        is_padding = (item_seq == 0)
        # Drop logic: drop if (prob < ratio) AND (not padding)
        drop_mask = (keep_probs < ratio) & (~is_padding)
        
        # Sort scores:
        # Kept items: score = index
        # Dropped items: score = index + seq_len
        # Padding: score = index + 2*seq_len (Always last)
        
        sort_scores = indices.clone()
        sort_scores[drop_mask] += seq_len
        sort_scores[is_padding] += 2 * seq_len
        
        # Sort to get new order
        _, sorted_indices = torch.sort(sort_scores, dim=1)
        
        # Gather
        augmented_seq = torch.gather(item_seq, 1, sorted_indices)
        
        # The result:
        # [A, B, C, D, 0] -> Drop B
        # Scores: A:0, B:1+5=6, C:2, D:3, 0:14
        # Sorted indices: 0(A), 2(C), 3(D), 1(B), 4(0)
        # Result: [A, C, D, B, 0]
        # Wait, B is at the end but it's not 0! It's 'B'.
        # We need to mask the dropped items at the end to 0.
        
        # Determine effective new length?
        # Or simpler:
        # Just mask the 'dropped' items in the gathered sequence?
        # How to know which were dropped?
        # The 'dropped' items are now at positions [new_len : old_len_without_pad]
        # The 'padding' items are at [old_len_without_pad : ]
        
        # Actually, simpler way:
        # Perform the sort.
        # Then, apply a mask based on the 'sorted' drop status?
        # Or faster:
        # Just mask the sources first?
        # If I map dropped items to 0 (padding) BEFORE gather?
        # [A, 0(was B), C, D, 0]
        # Scores: A:0, B(0):6, C:2, D:3, 0:14
        # Sorted: 0->A, 2->C, 3->D, 1->0, 4->0
        # Result: [A, C, D, 0, 0]
        # Correct!
        
        # So: 
        # 1. Identify dropped items.
        # 2. Compute sort indices based on drop status.
        # 3. Set dropped items to 0 in original.
        # 4. Gather.
        
        temp_seq = item_seq.clone()
        temp_seq[drop_mask] = 0
        
        augmented_seq = torch.gather(temp_seq, 1, sorted_indices)
        
        return augmented_seq

    def _item_reorder(self, item_seq, item_seq_len):
        """
        Vectorized Reorder (Shuffle Subsequence):
        Shuffle a random contiguous window? 
        Or just shuffle random items?
        'Reorder' usually means shuffling a subsequence.
        Vectorizing contiguous window selection is hard.
        
        Relaxed implementation: Randomly shuffle a subset of items.
        """
        batch_size, seq_len = item_seq.shape
        ratio = self.aug_ratio
        
        # Select items to shuffle
        # shuffle_mask = (prob < ratio) & (not padding)
        probs = torch.rand_like(item_seq, dtype=torch.float)
        is_padding = (item_seq == 0)
        shuffle_mask = (probs < ratio) & (~is_padding)
        
        # We want to shuffle the positions of items in 'shuffle_mask'.
        # Keep others fixed.
        
        # Indices: [0, 1, 2...]
        indices = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1).float()
        
        # Add noise to the indices of shuffled items
        # If we just add random noise to their sort score?
        # Fixed items: score = index * Large_Scale
        # Shuffled items: score = index * Large_Scale + Random(-Window, +Window)?
        # No, that just jitters local correlations.
        
        # True shuffle:
        # Give them random scores in a specific range?
        # This is strictly harder to maintain "shuffled subset in place".
        
        # Simplified Reorder:
        # Just shuffle ALL non-padding items? No, that's too aggressive.
        # Shuffle a "Block"?
        
        # Let's fallback to "Random Noise Sort" for the whole sequence?
        # "Add random noise to position indices"
        # preserve_mask = prob > ratio
        # scores = indices.clone()
        # scores[~preserve_mask] = rand() * seq_len (random positions)
        # scores[preserve_mask] = indices (keep original order relative to each other?)
        # This is a bit chaotic.
        
        # BETTER FALLBACK for speed:
        # Just use _item_crop (Subsample) or _item_mask (Drop) for now.
        # Reordering is less critical than Crop/Mask for SBR (sequential dependency is key).
        # Actually, let's just use Mask and Crop for efficiency if Reorder is hard.
        # BUT user asked for all 3.
        
        # Vectorized Shuffle:
        # 1. Identify padding
        # 2. Add random noise to ALL valid indices?
        #    This destroys ALL order.
        # 3. Add random noise to SOME valid indices?
        #    Swap them around.
        
        noise = torch.rand_like(indices) * 0.1 # Small noise
        # If we want to reorder specific items:
        # We need their new positions to be random relative to each other.
        
        # Strategy:
        # augment() chooses ONE method.
        # If Reorder is chosen (33%):
        # We can accept a slightly slower implementation or a "Global Shuffle" of a subset?
        # Let's do: Shuffle valid items locally.
        # scores = indices + (rand - 0.5) * ratio * seq_len
        # Valid items only.
        
        scores = indices.clone()
        perturbation = (torch.rand_like(scores) - 0.5) * (ratio * seq_len) # Noise proportional to ratio
        
        # Apply perturbation only to valid items
        mask = (~is_padding)
        scores[mask] += perturbation[mask]
        
        # Padding stays at infinity
        scores[is_padding] = 1e9
        
        _, sorted_indices = torch.sort(scores, dim=1)
        augmented_seq = torch.gather(item_seq, 1, sorted_indices)
        
        return augmented_seq

    def augment(self, item_seq, item_seq_len):
        """
        Apply random augmentation.
        Now fully vectorized.
        Using a batch-level decision (faster) or item-level?
        Item-level switch is hard to vectorize if branches differ structurally.
        
        Speed compromise:
        Choose ONE augmentation type for the WHOLE batch randomly.
        This is statistically fine for training (batches are random).
        """
        # Switch: 0=Crop, 1=Mask, 2=Reorder
        switch = torch.rand(1).item()
        
        if switch < 0.33:
            aug_seq = self._item_crop(item_seq, item_seq_len)
        elif switch < 0.66:
            aug_seq = self._item_mask(item_seq, item_seq_len)
        else:
            aug_seq = self._item_reorder(item_seq, item_seq_len)
            
        # Safety Check: Prevent all-zero sequences (causes NaN in F.normalize)
        # Check if row sum is 0 (assuming item_ids > 0)
        # If so, revert to original
        is_empty = (aug_seq.sum(dim=1) == 0)
        if is_empty.any():
            aug_seq[is_empty] = item_seq[is_empty]
            
        return aug_seq
