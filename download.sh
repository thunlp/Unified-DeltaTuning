wget https://thunlp.oss-cn-qingdao.aliyuncs.com/thunlp_unified_PETs_data_full_new_upload.tar.gz
tar -zxvf thunlp_unified_PETs_data_full_new_upload.tar.gz data

wget https://thunlp.oss-cn-qingdao.aliyuncs.com/thunlp_unified_PETs_pretrained_models_upload.tar.gz
tar -zxvf thunlp_unified_PETs_pretrained_models_upload.tar.gz pretrained_models

mkdir models
cd models
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/thunlp_unified_PETs_full_data_adapter.tar.gz
tar -zxvf thunlp_unified_PETs_full_data_adapter.tar.gz full_data_adapter
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/thunlp_unified_PETs_full_data_lora.tar.gz
tar -zxvf thunlp_unified_PETs_full_data_lora.tar.gz full_data_lora
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/thunlp_unified_PETs_full_data_prefix.tar.gz
tar -zxvf thunlp_unified_PETs_full_data_prefix.tar.gz full_data_prefix