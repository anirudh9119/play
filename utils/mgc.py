import pipes, os, subprocess, tempfile
import numpy as np

def wav2mgcf0(x, order=34, frame_window=512, zerofill_width=1024, shift_window=64, pass_const=0.4, min_pitch=20, max_pitch=500, mgcep_gamma=2):
    # Convert from int to float32, but keep numbers as integers
    x = x.astype('float32')

    # Compute pitchmgc
    pitch_cmd = 'pitch -a 0 -s 16 -p {} -L {} -H {}'.format(shift_window, min_pitch, max_pitch)

    p = subprocess.Popen(pitch_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    stdout, stderr = p.communicate(x.tobytes())
    f0 = np.fromstring(stdout, dtype='float32')

    # Compute MGC coefficients
    mgc_cmd = 'frame -l {} -p {} | window -l {} -L {} | mgcep -m {} -a {} -c {} -l {} -e 0.0012'.format(frame_window, shift_window, frame_window, zerofill_width, order, pass_const, mgcep_gamma, zerofill_width)

    p = subprocess.Popen(mgc_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    stdout, stderr = p.communicate(x.tobytes())
    mgc = np.fromstring(stdout, dtype='float32').reshape((len(f0), order+1))
    return mgc, f0

def mgcf02wav(mgc, f0, order=34, shift_window=64, pass_const=0.4, mgcep_gamma=2, gaussian=False):
    mgc = mgc.astype('float32')
    f0 = f0.astype('float32')

    with tempfile.NamedTemporaryFile() as f:
        mgc_fix_cmd = 'mgc2mgclsp -m {} -a {} -c {} -s 16000 | lspcheck -m {} -c -r 0.01 | mgclsp2mgc -m {} -a {} -c {}'.format(order, pass_const, mgcep_gamma, order, order, pass_const, mgcep_gamma)

        p = subprocess.Popen(mgc_fix_cmd, stdout=f, stdin=subprocess.PIPE, shell=True)
        mgc_fix, stderr = p.communicate(mgc.ravel().tobytes())
        f.file.flush()
        f.file.close()

        excitation_cmd = 'excite -p {}'.format(shift_window)
        p = subprocess.Popen(excitation_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
        excitation, stderr = p.communicate(f0.tobytes())

        synthesis_cmd = 'mglsadf -m {} -a {} -c {} -p {} {}'.format(order, pass_const, mgcep_gamma, shift_window, f.name)
        p = subprocess.Popen(synthesis_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
        stdout, stderr = p.communicate(excitation)
        y = np.fromstring(stdout, dtype='float32')

    return y

if __name__ == '__main__':
    from scipy.io import wavfile
    import sys

    fs, x = wavfile.read(sys.argv[1])
    mgc, f0 = wav2mgcf0(x)

    x_synth = mgcf02wav(mgc_with_noise, f0_with_noise)

    # Normalize x_synth by its maximum to avoid clipping
    x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15

    wavfile.write('test_mgcf.wav', 16000, x_synth.astype('int16'))