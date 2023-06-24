#!/bin/sh
python -u get_entity_embedding.py
python -u get_gnn_embedding.py
python -u get_muti_view_interaction_feature.py
python -u output_model.py
