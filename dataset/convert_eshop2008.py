import argparse
import csv
import datetime
import os


def read_sessions(csv_path):
    sess_clicks = {}
    sess_date = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sessid = row["session ID"].strip()
            item = row["page 2 (clothing model)"].strip()
            year = int(row["year"])
            month = int(row["month"])
            day = int(row["day"])
            order = int(row["order"])
            ts = datetime.datetime(year, month, day).timestamp()

            if sessid not in sess_clicks:
                sess_clicks[sessid] = []
            sess_clicks[sessid].append((order, item))
            sess_date[sessid] = ts

    for sessid in sess_clicks:
        sess_clicks[sessid].sort(key=lambda x: x[0])
        sess_clicks[sessid] = [x[1] for x in sess_clicks[sessid]]

    return sess_clicks, sess_date


def filter_sessions(sess_clicks, sess_date):
    for s in list(sess_clicks):
        if len(sess_clicks[s]) <= 1:
            del sess_clicks[s]
            del sess_date[s]

    iid_counts = {}
    for s, seq in sess_clicks.items():
        for iid in seq:
            iid_counts[iid] = iid_counts.get(iid, 0) + 1

    for s in list(sess_clicks):
        filseq = [i for i in sess_clicks[s] if iid_counts[i] >= 5]
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq


def split_sessions(sess_date):
    dates = list(sess_date.items())
    dates.sort(key=lambda t: t[1])

    tot = len(dates)
    train_split = int(tot * 0.8)
    valid_split = int(tot * 0.9)

    train_dates = dates[:train_split]
    valid_dates = dates[train_split:valid_split]
    test_dates = dates[valid_split:]
    return train_dates, valid_dates, test_dates


def obtain_train(train_sess, sess_clicks):
    item_dict = {}
    item_ctr = 1

    train_dates = []
    train_seqs = []
    for s, date in train_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i not in item_dict:
                item_dict[i] = item_ctr
                item_ctr += 1
            outseq.append(item_dict[i])
        if len(outseq) >= 2:
            train_dates.append(date)
            train_seqs.append(outseq)
    return train_dates, train_seqs, item_dict


def obtain_eval(eval_sess, sess_clicks, item_dict):
    eval_dates = []
    eval_seqs = []
    for s, date in eval_sess:
        seq = sess_clicks[s]
        outseq = [item_dict[i] for i in seq if i in item_dict]
        if len(outseq) >= 2:
            eval_dates.append(date)
            eval_seqs.append(outseq)
    return eval_dates, eval_seqs


def process_seqs(iseqs, idates, max_seq_len=50):
    out_seqs = []
    out_dates = []
    labs = []
    for seq, date in zip(iseqs, idates):
        for i in range(1, len(seq)):
            labs.append(seq[-i])
            out_seqs.append(seq[:-i][-max_seq_len:])
            out_dates.append(date)
    return out_seqs, out_dates, labs


def write_inter(path, seqs, labels, start_idx):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(["session_id:token", "item_id_list:token_seq", "item_id:token"]) + "\n")
        for i, (seq, lab) in enumerate(zip(seqs, labels), start=1):
            f.write(f"{start_idx + i}\t{' '.join(map(str, seq))}\t{lab}\n")
    return start_idx + len(seqs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="e-shop clothing 2008.csv",
        help="Path to e-shop clothing 2008 csv file",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/eshop2008",
        help="Output dataset directory",
    )
    args = parser.parse_args()

    sess_clicks, sess_date = read_sessions(args.input)
    filter_sessions(sess_clicks, sess_date)
    train_sess, valid_sess, test_sess = split_sessions(sess_date)

    tr_dates, tr_seqs, item_dict = obtain_train(train_sess, sess_clicks)
    va_dates, va_seqs = obtain_eval(valid_sess, sess_clicks, item_dict)
    te_dates, te_seqs = obtain_eval(test_sess, sess_clicks, item_dict)

    tr_seqs, _, tr_labs = process_seqs(tr_seqs, tr_dates)
    va_seqs, _, va_labs = process_seqs(va_seqs, va_dates)
    te_seqs, _, te_labs = process_seqs(te_seqs, te_dates)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = os.path.basename(args.output_dir.rstrip("/"))

    idx = 0
    idx = write_inter(os.path.join(args.output_dir, f"{dataset_name}.train.inter"), tr_seqs, tr_labs, idx)
    idx = write_inter(os.path.join(args.output_dir, f"{dataset_name}.valid.inter"), va_seqs, va_labs, idx)
    idx = write_inter(os.path.join(args.output_dir, f"{dataset_name}.test.inter"), te_seqs, te_labs, idx)

    print(f"Created dataset: {dataset_name}")
    print(f"Interactions: {idx}")
    print(f"Train/Valid/Test examples: {len(tr_seqs)}/{len(va_seqs)}/{len(te_seqs)}")
    print(f"Mapped training items: {len(item_dict)}")


if __name__ == "__main__":
    main()
