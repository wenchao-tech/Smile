# Main table
CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_name besk.ckpt --test_dir ./data_awracle/RESIDE/ --in_context_dir ./data_awracle/Train/Dehaze/ --test_json dehaze_reside_test.json --output_path ./reside/ --in_context_file dehaze_reside_train.json 

CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_name besk.ckpt --test_dir ./data_awracle/Rain13K/ --in_context_dir ./data_awracle/Train/Derain/ --test_json derain_test_rain100l.json --output_path ./rain100l/ --in_context_file derain_train.json 

CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_name besk.ckpt --test_dir ./data_awracle/Rain13K/ --in_context_dir ./data_awracle/Train/Derain/ --test_json derain_test_rain100h.json --output_path ./rain100h/ --in_context_file derain_train.json 

CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_name besk.ckpt --test_dir ./data_awracle/Snow100k/ --in_context_dir ./data_awracle/Train/Desnow/ --test_json desnow_snow100_L_test.json --output_path ./snowL/ --in_context_file desnow_snow100_train.json 

CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_name besk.ckpt --test_dir ./data_awracle/Snow100k/ --in_context_dir ./data_awracle/Train/Desnow/ --test_json desnow_snow100_M_test.json --output_path ./snowM/ --in_context_file desnow_snow100_train.json 

CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_name besk.ckpt --test_dir ./data_awracle/Snow100k/ --in_context_dir ./data_awracle/Train/Desnow/ --test_json desnow_snow100_S_test.json --output_path ./snowS/ --in_context_file desnow_snow100_train.json 

# Mixed degradation on CSD: Snow + haze
CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_name besk.ckpt --test_dir ./data_awracle/CSD/ --in_context_dir ./data_awracle/Train/Desnow/ --test_json desnow_csd_Test.json --output_path ./csd_desnow/ --in_context_file desnow_snow100_train.json 

CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_name besk.ckpt --test_dir ./data_awracle/CSD/ --in_context_dir ./data_awracle/Train/Dehaze/ --test_json desnow_csd_Test.json --output_path ./csd_dehaze/ --in_context_file dehaze_reside_train.json 
