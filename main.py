import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed, set_color


from core_ave import COREave
from core_trm import COREtrm
from improvments.core_trm_enhanced import COREtrmEnhanced
from improvments.core_trm_dual_attention import COREtrmDualAttention
from improvments.core_trm_enhanced_pe import COREtrmEnhancedPE
from improvments.core_trm_hard_negatives import COREtrmHardNeg
from improvments.core_trm_contrastive import COREtrmContrastive


def run_single_model(args):
    # configurations initialization
    if args.model == 'ave':
        model_class = COREave
    elif args.model == 'trm':
        model_class = COREtrm
    elif args.model == 'trm_enhanced':
        model_class = COREtrmEnhanced
    elif args.model == 'trm_dual_attention':
        model_class = COREtrmDualAttention
    elif args.model == 'trm_enhanced_pe':
        model_class = COREtrmEnhancedPE
    elif args.model == 'trm_hard_negatives':
        model_class = COREtrmHardNeg
    elif args.model == 'trm_contrastive':
        model_class = COREtrmContrastive
    else:
        model_class = COREtrm
    
    config_file = f'props/core_{args.model}.yaml' if args.model in ['ave', 'trm', 'trm_enhanced', 'trm_dual_attention', 'trm_enhanced_pe', 'trm_hard_negatives', 'trm_contrastive'] else 'props/core_trm.yaml'
    
    config = Config(
        model=model_class,
        dataset=args.dataset,
        config_file_list=['props/overall.yaml', config_file]
    )
    # Apply optional runtime overrides from CLI
    if getattr(args, 'temperature', None) is not None:
        config['temperature'] = float(args.temperature)
    if getattr(args, 'item_dropout', None) is not None:
        config['item_dropout'] = float(args.item_dropout)
    if getattr(args, 'hard_neg_weight', None) is not None:
        config['hard_neg_weight'] = float(args.hard_neg_weight)
    if getattr(args, 'use_hard_negatives', None) is not None:
        config['use_hard_negatives'] = args.use_hard_negatives

    # If we run a dropout sweep, iterate over specified rhos and override config values
    def run_one(cfg, args):
        init_seed(cfg['seed'], cfg['reproducibility'])
        init_logger(cfg)
        logger = getLogger()

        logger.info(cfg)

        dataset = create_dataset(cfg)
        logger.info(dataset)

        train_data, valid_data, test_data = data_preparation(cfg, dataset)

        if args.model == 'ave':
            model = COREave(cfg, train_data.dataset).to(cfg['device'])
        elif args.model == 'trm':
            model = COREtrm(cfg, train_data.dataset).to(cfg['device'])
        elif args.model == 'trm_enhanced':
            model = COREtrmEnhanced(cfg, train_data.dataset).to(cfg['device'])
        elif args.model == 'trm_dual_attention':
            model = COREtrmDualAttention(cfg, train_data.dataset).to(cfg['device'])
        elif args.model == 'trm_enhanced_pe':
            model = COREtrmEnhancedPE(cfg, train_data.dataset).to(cfg['device'])
        elif args.model == 'trm_hard_negatives':
            model = COREtrmHardNeg(cfg, train_data.dataset).to(cfg['device'])
        elif args.model == 'trm_contrastive':
            model = COREtrmContrastive(cfg, train_data.dataset).to(cfg['device'])
        else:
            raise ValueError('model can only be "ave", "trm", "trm_enhanced", "trm_dual_attention", "trm_enhanced_pe", "trm_hard_negatives", or "trm_contrastive".')
        logger.info(model)

        trainer = get_trainer(cfg['MODEL_TYPE'], cfg['model'])(cfg, model)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=cfg['show_progress']
        )

        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=cfg['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': cfg['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }

    # If sweep flag set and using COREtrm on diginetica, sweep item_dropout values
    if getattr(args, 'sweep_dropout', False) and args.model == 'trm' and args.dataset == 'diginetica':
        # target dropout ratios to test
        rhos = [0.15, 0.20, 0.25, 0.30]
        results = []
        for rho in rhos:
            # override crucial hyperparameters for consistency
            config['item_dropout'] = rho
            config['temperature'] = 0.07  # set temperature tau to 0.07 as requested
            # run training/eval for this config
            res = run_one(config, args)
            # attach rho for reporting
            res['rho'] = rho
            results.append(res)

        # Print concise summary of R@20 and MRR@20 per rho
        logger = getLogger()
        logger.info('Dropout sweep results:')
        for r in results:
            tr = r['test_result']
            # try to extract Recall@20 and MRR@20 from the returned metrics dict
            recall_key = None
            mrr_key = None
            for k in tr.keys():
                lk = k.lower()
                if 'recall' in lk and '@20' in lk:
                    recall_key = k
                if 'mrr' in lk and '@20' in lk:
                    mrr_key = k
            recall = tr.get(recall_key, 'N/A')
            mrr = tr.get(mrr_key, 'N/A')
            logger.info(f"rho={r['rho']}: R@20={recall}, MRR@20={mrr}")

        return results

    else:
        # default single run
        return run_one(config, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='trm', help='ave, trm, trm_enhanced, trm_dual_attention, trm_enhanced_pe, trm_hard_negatives, or trm_contrastive')
    parser.add_argument('--dataset', type=str, default='diginetica', help='diginetica, nowplaying, retailrocket, tmall, yoochoose')
    parser.add_argument('--sweep-dropout', dest='sweep_dropout', action='store_true', help='If set, sweep item_dropout over predefined rhos')
    parser.add_argument('--temperature', type=float, default=None, help='Override temperature (tau) in config')
    parser.add_argument('--item-dropout', type=float, default=None, help='Override item_dropout in config')
    parser.add_argument('--hard-neg-weight', type=float, default=None, help='Override hard negative weight')
    parser.add_argument('--use-hard-negatives', type=lambda x: x.lower() == 'true', default=None, help='Enable/disable hard negatives (true/false)')
    args, _ = parser.parse_known_args()

    # Pass through any runtime overrides to the run function
    run_single_model(args)
