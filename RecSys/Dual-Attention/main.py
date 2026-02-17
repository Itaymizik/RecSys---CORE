import argparse
import os
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed, set_color

from core_ave import COREave
from core_trm import COREtrm, COREtrmDualAttention


MODEL_REGISTRY = {
    'ave': (COREave, 'core_ave.yaml'),
    'trm': (COREtrm, 'core_trm.yaml'),
    'trm_da': (COREtrmDualAttention, 'core_trm_da.yaml'),
}


def run_single_model(args):
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Use one of: {', '.join(MODEL_REGISTRY)}")

    model_class, model_cfg = MODEL_REGISTRY[args.model]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    config = Config(
        model=model_class,
        dataset=args.dataset,
        config_file_list=[
            os.path.join(base_dir, 'props', 'overall.yaml'),
            os.path.join(base_dir, 'props', model_cfg),
        ],
        config_dict={'train_neg_sample_args': None},
    )

    if args.temperature is not None:
        config['temperature'] = float(args.temperature)

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config['show_progress'],
    )
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='trm_da', help='ave, trm, trm_da')
    parser.add_argument('--dataset', type=str, default='diginetica', help='diginetica, nowplaying, retailrocket, tmall, yoochoose')
    parser.add_argument('--temperature', type=float, default=None, help='Override temperature')
    args, _ = parser.parse_known_args()

    run_single_model(args)
