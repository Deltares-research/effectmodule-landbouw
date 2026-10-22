def write_outputs(median_actual_masked, median_derv_masked, act_med_fn, derv_med_fn):
    median_actual_masked.rio.to_raster(act_med_fn, nodata=-9999)
    median_derv_masked.rio.to_raster(derv_med_fn, nodata=-9999)
