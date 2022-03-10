import torch as th
import torch.nn.functional as tnf


def mse(est, target):
    return tnf.mse_loss(est, target)


def mae(est, target):
    return tnf.l1_loss(est, target)


def gccrn_mse(model, est_specs, target):
    real_t, imag_t = model.stft.transform(target)
    specs_t = th.cat([real_t, imag_t], dim=1)
    return mse(est_specs, specs_t)



def compress_magphase(real, imag, alpha):
    compressed_mags = th.pow(real**2 + imag**2 + 1e-8, alpha*0.5)
    phase = th.atan2(imag, real+1e-8)
    real = compressed_mags * th.cos(phase)
    imag = compressed_mags * th.sin(phase)
    compressed_cspec = th.cat([real, imag], dim=1)
    return compressed_mags, compressed_cspec


def plcpa(model, est, label):
    real_est, imag_est = th.chunk(est["cspec"], 2, dim=1)
    com_mags_est, com_cspec_est = compress_magphase(real_est, imag_est, 0.5)
    real_t, imag_t = model.stft.transform(label)
    # Reomve DC or not, You may uncomment this...
    # real_t[:, 0, :] = 0
    # imag_t[:, 0, :] = 0
    com_mags_t, com_cspec_t = compress_magphase(real_t, imag_t, 0.5)
    loss_amp = mse(com_mags_est, com_mags_t)
    loss_pha = mse(com_cspec_est, com_cspec_t)
    # wav_mse = mae(est["wav"], label)
    return loss_amp + loss_pha


def echo_weight(r_t, i_t, r_e, i_e):
    r_t_s = th.square(r_t)
    i_t_s = th.square(i_t)
    r_e_s = th.square(r_e)
    i_e_s = th.square(i_e)
    mag_s_2 = r_t_s + i_t_s
    mag_f_2 = r_e_s + i_e_s
    mag_weight = th.divide(mag_f_2, mag_f_2 + mag_s_2 + 1e-8) + 1
    return mag_weight


def weighted_plcpa(stft, est, label, e):
    """
    stft: as named, STFT function.
    est: shape Batch x 2F x T.
    label, e: Batch x Nsamples.
    e: echo
    """
    real_est, imag_est = th.chunk(est, 2, dim=1)
    com_mags_est, com_cspec_est = compress_magphase(real_est, imag_est, 0.5)
    real_t, imag_t = stft.transform(label)
    real_e, imag_e = stft.transform(e)
    # Reomve DC or not, You may uncomment this...
    # real_t[:, 0, :] = 0.0
    # imag_t[:, 0, :] = 0.0
    # real_e[:, 0, :] = 0.0
    # imag_e[:, 0, :] = 0.0
    com_mags_t, com_cspec_t = compress_magphase(real_t, imag_t, 0.5)
    mag_weight = echo_weight(real_t, imag_t, real_e, imag_e)
    loss_amp = th.mean(th.square(com_mags_est-com_mags_t)*mag_weight)
    loss_pha = mse(com_cspec_est, com_cspec_t)
    return loss_amp + loss_pha
  
  
  def weighted_plcpa_ce(stft, est, label, e, vad_label):
    """
    Final loss used in the paper.
    
    params:
    stft: as named, STFT function.
    est[1]: estimated complex spectrum, Batch x 2F x T.
    est[2]: masked complex spectrum, as described in Eq. (8)
    est[3]: VAD estimation, shape B x T x 2
    label, e: shape Batch x Nsamples.
    e: echo
    """
    real_est, imag_est = th.chunk(est[1], 2, dim=1)
    com_mags_est, com_cspec_est = compress_magphase(real_est, imag_est, 0.5)
    real_t, imag_t = stft.transform(label)
    real_e, imag_e = stft.transform(e)
    # Reomve DC or not, You may uncomment this...
    # real_t[:, 0, :] = 0.0
    # imag_t[:, 0, :] = 0.0
    # real_e[:, 0, :] = 0.0
    # imag_e[:, 0, :] = 0.0
    com_mags_t, com_cspec_t = compress_magphase(real_t, imag_t, 0.5)
    mag_weight = echo_weight(real_t, imag_t, real_e, imag_e)
    loss_amp = th.mean(th.square(com_mags_est-com_mags_t)*mag_weight)
    loss_pha = mse(com_cspec_est, com_cspec_t)
    masked_mse_loss = mse(est[2], th.cat([real_t, imag_t], dim=1))
    ce = tnf.cross_entropy(est[3], vad_label.long())
    return loss_amp + loss_pha + 0.2*masked_mse_loss + 0.1*ce
