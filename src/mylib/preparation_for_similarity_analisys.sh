source VENV/bin/activate
export OS_CUDA_VISIBLE_DEVICES="0"
python3 parse_multiwoz.py
python3 sentence_encoding.py
python3 dgac_clustering.py
python3 assign_clust_tf_idf_names.py