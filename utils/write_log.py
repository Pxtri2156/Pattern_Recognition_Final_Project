import logging

def write_config_args_2_log(cfgs, args, logger):
    logger.info("ARGUMENT ")
    logger.info("{} The path root of exacted features: {}".format('#'*3, args.root))
    logger.info("{} Dataset name to train: {}".format('#'*3, args.data_n))
    logger.info("{} The methods exacted feature: {}".format('#'*3, args.features))
    logger.info("{} Augment data: {}".format('#'*3, args.argu))
    logger.info("{} Dataset name to train: {}".format('#'*3, args.data_n))
    logger.info("{} The path of model ouput : {}".format('#'*3, args.output))
    logger.info("{} The path of configs file : {}".format('#'*3, args.configs_file))