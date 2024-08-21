'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de

code updated by Wiam Boumaazi, wboumaazi@gmail.com

'''


import os
import cv2
import scipy
import tempfile
import numpy as np
import tensorflow as tf
from subprocess import call
from scipy.io import wavfile
from psbody.mesh import Mesh
from utils.audio_handler import  AudioHandler
from utils.rendering import render_mesh_helper
import time
import multiprocessing
from pydub import AudioSegment
from pydub.utils import make_chunks

def process_audio(ds_path, audio, sample_rate,l): 
    config = {}
    config['deepspeech_graph_fname'] = ds_path
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29
    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1

    tmp_audio = {'subj': {'seq': {'audio': audio, 'sample_rate': sample_rate}}}
    audio_handler = AudioHandler(config)
    row = l[0]
    row.append(audio_handler.process(tmp_audio)['subj']['seq']['audio'])
    l[0] = row
    
def output_sequence_meshes(sequence_vertices, template, out_path, uv_template_fname='', texture_img_fname=''):
    mesh_out_path = os.path.join(out_path, 'meshes')
    if not os.path.exists(mesh_out_path):
        os.makedirs(mesh_out_path)

    if os.path.exists(uv_template_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
    else:
        vt, ft = None, None

    num_frames = sequence_vertices.shape[0]
    for i_frame in range(num_frames):
        out_fname = os.path.join(mesh_out_path, '%05d.obj' % i_frame)
        out_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            out_mesh.vt, out_mesh.ft = vt, ft
        if os.path.exists(texture_img_fname):
            out_mesh.set_texture_image(texture_img_fname)
        out_mesh.write_obj(out_fname)

def render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, uv_template_fname='', texture_img_fname=''):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 60, (800, 800), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (800, 800), True)

    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
    else:
        vt, ft = None, None
        tex_img = None

    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    for i_frame in range(num_frames):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
        writer.write(img)
    writer.release()

    video_fname = os.path.join(out_path, 'video.mp4')
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)

def multi_run_wrapper(args):
   return process_audio(*args)

def inference(tf_model_fname, ds_fname, audio_fname, template_fname, number_processes,condition_idx, out_path, render_sequence=True, uv_template_fname='', texture_img_fname=''):
    # "zero pose" template mesh in FLAME topology to be animated
    template = Mesh(filename=template_fname)
    myaudio = AudioSegment.from_file(audio_fname)
    chunks = make_chunks(myaudio, (len(myaudio)/number_processes))


    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")
   
    #PART1
    chunk1 = "/home/wiam/Desktop/voca/chunk0.wav"
    chunk2 = "/home/wiam/Desktop/voca/chunk1.wav"
    chunk3 = "/home/wiam/Desktop/voca/chunk2.wav"
    chunk4 = "/home/wiam/Desktop/voca/chunk3.wav"
    #chunk5 = "/home/wiam/Desktop/voca/one_page/5/4.wav"
    #chunk6 = "/home/wiam/Desktop/temp_voca/1750/6/5.wav"
    #chunk7 = "/home/wiam/Desktop/temp_voca/272/7/6.wav"
    #chunk8 = "/home/wiam/Desktop/temp_voca/272/6/5.wav"
    ###PART1 END

    #PART1
    sample_rate1, audio1 = wavfile.read(chunk1)
    sample_rate2, audio2 = wavfile.read(chunk2)
    sample_rate3, audio3 = wavfile.read(chunk3)
    sample_rate4, audio4 = wavfile.read(chunk4)
    #sample_rate5, audio5 = wavfile.read(chunk5)
    #sample_rate6, audio6 = wavfile.read(chunk6)
    #sample_rate7, audio7 = wavfile.read(chunk7)
    #sample_rate6, audio6 = wavfile.read(chunk6)
    #PART2 END
    
    
    manager = multiprocessing.Manager()
    lst1 = manager.list()
    lst2= manager.list()
    lst3 = manager.list()
    lst4 = manager.list()
    lst5 = manager.list()
    lst6 = manager.list()
    lst7 = manager.list()
    lst6 = manager.list()

    lst1.append([1])
    lst2.append([2])
    lst3.append([3])
    lst4.append([4])
    lst5.append([5])
    lst6.append([6])
    lst7.append([7])
    lst6.append([6])
    
    #PART3
    p1 = multiprocessing.Process(target=process_audio, args=(ds_fname, audio1, sample_rate1, lst1,))# lock))
    p2 = multiprocessing.Process(target=process_audio, args=(ds_fname, audio2, sample_rate2, lst2,)) #lock))
    p3 = multiprocessing.Process(target=process_audio, args=(ds_fname, audio3, sample_rate3, lst3,))# lock))
    p4 = multiprocessing.Process(target=process_audio, args=(ds_fname, audio4, sample_rate4, lst4,))
    #p5 = multiprocessing.Process(target=process_audio, args=(ds_fname, audio5, sample_rate5, lst5,))
    #p6 = multiprocessing.Process(target=process_audio, args=(ds_fname, audio6, sample_rate6, lst6,))
    #p7 = multiprocessing.Process(target=process_audio, args=(ds_fname, audio7, sample_rate7, lst7,))
    #p6 = multiprocessing.Process(target=process_audio, args=(ds_fname, audio6, sample_rate6, lst6,))
    
    start = time.perf_counter()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    #p5.start()
    #p6.start()
    #p7.start()
    #p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    #p5.join()
    #p6.join()
    #p7.join()
    #p6.join()
    
    #PART3 END

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} seconds')

    #PART4
    ch1 = np.array(lst1[0][1])
    ch2 = np.array(lst2[0][1])
    ch3 = np.array(lst3[0][1])
    ch4 = np.array(lst4[0][1])
    #ch5 = np.array(lst5[0][1])
    #ch6 = np.array(lst6[0][1])
    #In case you chose to add more chunks or less chunks you need to change the line bellow
    concatenated_result = np.vstack((ch1,ch2,ch3,ch4)) 
    #PART4 END
    processed_audio = concatenated_result

    #Restart training from a saved graph and checkpoints.
    #Run inference from a saved graph and checkpoints.   
    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    condition_subject_id = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/condition_subject_id:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    input_template = graph.get_tensor_by_name(u'VOCA/Inputs_decoder/template_placeholder:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/output_decoder:0')

    num_frames = processed_audio.shape[0]
    feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                 condition_subject_id: np.repeat(condition_idx-1, num_frames),
                 is_training: False,
                 input_template: np.repeat(template.v[np.newaxis, :, :, np.newaxis], num_frames, axis=0)}

    with tf.Session() as session:
        # Restore trained model
        saver.restore(session, tf_model_fname)
        predicted_vertices = np.squeeze(session.run(output_decoder, feed_dict))
        output_sequence_meshes(predicted_vertices, template, out_path)
        if(render_sequence):
            render_sequence_meshes(audio_fname, predicted_vertices, template, out_path, uv_template_fname, texture_img_fname)
    tf.reset_default_graph()
    

def inference_interpolate_styles(tf_model_fname, ds_fname, audio_fname, template_fname, condition_weights, out_path):
    template = Mesh(filename=template_fname)

    sample_rate, audio = wavfile.read(audio_fname)
    


    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:, 0]

    processed_audio = process_audio(ds_fname, audio, sample_rate)

    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    condition_subject_id = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/condition_subject_id:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    input_template = graph.get_tensor_by_name(u'VOCA/Inputs_decoder/template_placeholder:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/output_decoder:0')

    non_zeros = np.where(condition_weights > 0.0)[0]
    condition_weights[non_zeros] /= sum(condition_weights[non_zeros])

    num_frames = processed_audio.shape[0]
    output_vertices = np.zeros((num_frames, template.v.shape[0], template.v.shape[1]))

    with tf.Session() as session:
        # Restore trained model
        saver.restore(session, tf_model_fname)

        for condition_id in non_zeros:
            feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                         condition_subject_id: np.repeat(condition_id, num_frames),
                         is_training: False,
                         input_template: np.repeat(template.v[np.newaxis, :, :, np.newaxis], num_frames, axis=0)}
            predicted_vertices = np.squeeze(session.run(output_decoder, feed_dict))
            output_vertices += condition_weights[condition_id] * predicted_vertices

        output_sequence_meshes(output_vertices, template, out_path)
