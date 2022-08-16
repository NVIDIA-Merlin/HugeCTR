set -e

# -------- dummy_var -------------- #
cd dummy_var
python dummy_var_handle_test.py
python dummy_var_initialize_test.py
python dummy_var_shape_test.py
python dummy_var_assign_test.py
python dummy_var_export_test.py
python dummy_var_sparse_read_test.py
python dummy_var_scatter_add_test.py
python dummy_var_scatter_update_test.py
cd ..

# -------- embedding collection -------------- #
cd embedding_collection
python preprocessing_forward_test.py
python lookup_forward_test.py
python lookup_forward_dynamic_test.py
python lookup_backward_test.py
python postprocessing_forward_test.py
python postprocessing_backward_test.py
cd ..

# -------- lookup -------------- #
cd lookup
python dist_select_test.py
python reorder_test.py
python gather_ex_test.py
python group_lookup_test.py
cd ..
