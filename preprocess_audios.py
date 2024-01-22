from email.mime import audio
import os
import sox

#pysox functions
def normalize_audio(standard_audio_path, recorded_audio_path, output_path=None):
    temp_path = 'temp.wav'
    tfm = sox.Transformer()
    #tfm.vad()
    tfm.build_file(recorded_audio_path, temp_path)
    standard = sox.file_info.duration(standard_audio_path)
    recorded = sox.file_info.duration(temp_path)
    tfm = sox.Transformer()
    tfm.tempo(recorded/standard, 's')
    if output_path == None:
        output_path = recorded_audio_path[:recorded_audio_path.rfind("_untrim.")] + ".wav"
    tfm.build_file(temp_path, output_path)
    print(f"Audio saved at {output_path}")
    os.system('rm temp.wav')
    
def removesilence(input_path, output_path):
    tfm = sox.Transformer()
    tfm.vad()
    tfm.build_file(input_path, output_path)

#converting all audios to .wav file
audio_dirs = ['preprocess_audios', 'preprocess_audios/standard']
for audio_dir in audio_dirs:
    for filename in [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]:
        raw_filename, raw_extension = os.path.splitext(filename)[0], os.path.splitext(filename)[1]
        if raw_extension != ".wav":
            input_path = os.path.join(audio_dir, filename)
            output_path = os.path.join(audio_dir, raw_filename + '.wav')
            try:
                os.system(f'ffmpeg -i {input_path} {output_path}')
            except:
                print("Skipping, cannot convert to .wav")
                continue
            else:
                os.system(f'rm {input_path}')

audio_dir = audio_dirs[0]
#pysox operations
for filename in [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]:
    print(f"Attempting to normalize {filename}")
    raw_filename, raw_extension = os.path.splitext(filename)[0], os.path.splitext(filename)[1]
    index = raw_filename.rfind("_")

    if(index == -1):
        print(f"File name \"{filename}\" invalid, format must be: [filename]_[standard].[extension] - skipping")
        continue
    
    standard_path = os.path.join(audio_dir, "standard", raw_filename[(index+1):] + '.wav')
    existing_path = os.path.join(audio_dir, raw_filename + '.wav')
    converted_path = os.path.join(audio_dir, "converted", raw_filename + '.wav')
    if os.path.exists(converted_path):
        print(f"Converted file already exists for {filename}, skipping")
    elif os.path.exists(standard_path):
        normalize_audio(standard_path, existing_path, converted_path)
    else:
        print(f"Standard for file \"{filename}\" doesn't exist, skipping")