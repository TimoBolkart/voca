import os

audio_dirs = ['preprocess_audios/converted', 'preprocess_audios/standard']
output_dir = 'generated_animations'

for audio_dir in audio_dirs:
    for filename in [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]:
        raw_filename, raw_extension = os.path.splitext(filename)[0], os.path.splitext(filename)[1]
        audio_path_wav = os.path.join(audio_dir, filename)
        output_path = os.path.join(output_dir, raw_filename)
        if os.path.exists(output_path):
            print(f"Vertices already exists for {filename}, skipping")
        else:
            os.system(f"python run_voca.py  --tf_model_fname './model/gstep_130700.model' --ds_fname './ds_graph/models/output_graph.pb' --audio_fname './{audio_path_wav}' --template_fname './template/FLAME_sample.ply' --condition_idx 3 --out_path './{output_path}' --visualize 'False'")
            print(f"Generated vertices for {filename}")