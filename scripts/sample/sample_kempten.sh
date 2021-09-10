python3 $HOME/CroScaleRep/sample.py \
--data $HOME/geo_loc_data/kempten/raw_telescope \
--data_type .tif \
--output $HOME/geo_loc_data_gen/kempten200 \
--microscope_src google_map_static_api_and_tile \
--telescope_src tiff \
--num_teles 200 \
--num_micros 32 \
--sample_mode random \
--gmap_api_key REPLACEWITHYOURKEY \
--url_secret  REPLACEWITHYOURURLSECRET \
--flip_axis 0 \
--scale_down 1