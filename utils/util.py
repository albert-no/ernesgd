import time

def path_generate(parser):
    file_path, mat_path, ckpt_path = 'temp_images', 'FID', 'ckpt'

    lr_str = str(parser.lr)
    b1_str = str(parser.b1)
    file_path += '_run_for_%d_epochs' % parser.n_epochs + '_FID_epochs_%d' % parser.FID_epochs + '_lr_%s' % lr_str + '_b1_%s' % b1_str
    mat_path += '_run_for_%d_epochs' % parser.n_epochs + '_FID_epochs_%d' % parser.FID_epochs + '_lr_%s' % lr_str + '_b1_%s' % b1_str
    ckpt_path += '_run_for_%d_epochs' % parser.n_epochs + '_FID_epochs_%d' % parser.FID_epochs + '_lr_%s' % lr_str + '_b1_%s' % b1_str


    if parser.simultaneous:
        file_path += '_sim'
        mat_path += '_sim'
        ckpt_path += '_sim'
    else:
        file_path += '_alt'
        mat_path += '_alt'
        ckpt_path += '_alt'

    if parser.anchoring:

        anchor_gamma_str = str(parser.anchor_gamma)
        anchor_gamma_str = 'p'.join(anchor_gamma_str.split('.'))

        file_path += '_anchor_' + parser.anchoring_decay_mode + '_anchor_start_from_epoch_%d' % parser.anchor_start_from + '_anchor_beta_%s' % anchor_gamma_str + '_anchor_update_period_%d' % parser.anchor_update_period
        mat_path += '_anchor_' + parser.anchoring_decay_mode + '_anchor_start_from_epoch_%d' % parser.anchor_start_from + '_anchor_beta_%s' % anchor_gamma_str + '_anchor_update_period_%d' % parser.anchor_update_period
        ckpt_path += '_anchor_' + parser.anchoring_decay_mode + '_anchor_start_from_epoch_%d' % parser.anchor_start_from + '_anchor_beta_%s' % anchor_gamma_str + '_anchor_update_period_%d' % parser.anchor_update_period

    if parser.optimism:
        anchor_gamma_str = str(parser.anchor_gamma)
        anchor_gamma_str = 'p'.join(anchor_gamma_str.split('.'))
        
        file_path += '_optimism' + '_optim_rate_%s' % anchor_gamma_str
        mat_path += '_optimism' + '_optim_rate_%s' % anchor_gamma_str
        ckpt_path += '_optimism' + '_optim_rate_%s' % anchor_gamma_str

    if parser.load_from_path is not None:
        file_path += '_loaded'
        mat_path += '_loaded'
        ckpt_path += '_loaded'

    time_stamp = time.strftime("_%Y%m%d_%H%M%S", time.localtime())
    seed_str = '_seed' + str(parser.seed)

    file_path += (seed_str + time_stamp)
    mat_path += (seed_str + time_stamp + '.mat')
    ckpt_path += (seed_str + time_stamp)

    return file_path, mat_path, ckpt_path
