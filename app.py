from speechbrain.pretrained import SepformerSeparation as separator
import torch
import torchaudio

model = separator.from_hparams(
    source="speechbrain/sepformer-wham-enhancement", 
    savedir='pretrained_models/sepformer-wham-enhancement', 
    run_opts={"device": "cuda"}
)

def infer(input_file: str, output_file: str = None):
    est_sources: torch.Tensor = model.separate_file(path=input_file)
    if output_file is None:
        return est_sources[:, :, 0]
    torchaudio.save(output_file, est_sources[:, :, 0].detach().cpu(), 44100)

infer("/media/alvynabranches/TOSHIBA EXT/03042022.mp3", "./output.wav")