
import os


import torch

from llava.mm_utils import get_model_base_from_path, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from EEG_Encoder.Tools.dataBuilder import CLIPDataset_Pair_Optimized, CLIPDataset_ThingEEG
from transformers.generation.streamers import TextStreamer
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import get_conversation_template

if os.environ.get('WaveMind_ROOT_PATH_') == None:
    raise RuntimeError("WaveMind_ROOT_PATH_ environment variable is not set. Please set it to the root path of the WaveMind project.")
else:
    project_root_path = os.environ.get('WaveMind_ROOT_PATH_')



model_path = f"{project_root_path}/EEGLLM/LLaVA/checkpoints/MELA/mela_0.6/mela_vicuna_7b_close_continue_lora"
root_path=f'{project_root_path}/Data_Engineering'

def select_data_for_test(dataset:str,batch_size:int=1):
    import os
    load_dir=os.path.join(os.environ['WaveMind_ROOT_PATH_'],'data/Total')
    if dataset!='THING-EEG':
        ds= CLIPDataset_Pair_Optimized(hdf5_file_path=os.path.join(load_dir,'data_label.h5'), mode='test',dataset_name=dataset,)
    else:
        ds=CLIPDataset_ThingEEG(train=False,train_Val_Same_set='none',model_type='ViT-L-14-336')
    return ds
        

def WaveMind_inference(model,tokenizer,question,processed_eeg,RAG=True,verbose=True,model_base=None):
    # Validate EEG shape against model's expected sampling rate
    expected_fs = model.config.eeg_sampling_rate
    if processed_eeg.shape[-1] != expected_fs:
        raise ValueError(
            f"EEG sampling rate mismatch: model expects {expected_fs}Hz "
            f"({expected_fs} samples), but got {processed_eeg.shape[-1]} samples. "
            f"Please preprocess EEG with correct sampling rate."
        )

    def get_RAG_search_result(eeg, llava_model,topk=None):
        topk=llava_model.DBtool.recommend_topk if topk is None else topk
        search_result=llava_model.DBtool.get_search_result_from_EEG(eeg_model=llava_model.get_model().get_neuro_tower(),eeg=eeg,topk=topk)
        return search_result

    inp = question.replace("<image>","")+(f"\n{get_RAG_search_result(processed_eeg,model)}" if RAG==True else '')+("<image>")

    conv = get_conversation_template(model_base).copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt',show_prompt=verbose).unsqueeze(0).to(model.device)


    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False) if verbose else None

    generate_kwargs ={

    }
    
    
    generate_kwargs['modilitys']=[processed_eeg]
    generate_kwargs['modility_types']=['eeg']
        



    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.5,
            max_new_tokens=200,
            streamer=streamer,
            use_cache=True,
            **generate_kwargs,
        )
        outputs = tokenizer.decode(output_ids[0]).strip()
        return outputs



if __name__ == '__main__':
    model_base=get_model_base_from_path(model_path)
    tokenizer, model, modility_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path),
    )
    ds=select_data_for_test(dataset='TUEV')
    
    sample=ds[0]
    category,eeg_tensor=sample['text'].decode(),sample['eeg_data']
    
    
    WaveMind_inference(model,tokenizer,category,eeg_tensor,RAG=True,verbose=True,model_base=model_base)
    
    
    
    
    
    
    
    
    