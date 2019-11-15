try:
    from mpi4py import *
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except:
    rank = 0
    n_ranks = 1
import os
import sys
import numpy as np
from teca import *


def get_padding_params(lat, lon):
    target_shape = 128 * np.ceil(lat/128.0)
    target_shape_diff = target_shape - lat
    padding_amount_lat = (int(np.ceil(target_shape_diff / 2.0)), int(np.floor(target_shape_diff / 2.0)))

    target_shape = 64 * np.ceil(lon/64.0)
    target_shape_diff = target_shape - lon
    padding_amount_lon = (int(np.ceil(target_shape_diff / 2.0)), int(np.floor(target_shape_diff / 2.0)))

    return padding_amount_lat, padding_amount_lon

def main():
    set_stack_trace_on_error()
    set_stack_trace_on_mpi_error()

    if (len(sys.argv) != 8):
        sys.stderr.write('\n\nUsage error:\n'\
            'test_deeplabv3p_ar_detect [deeplab_model] [resnet_model] ' \
            '[mesh data regex] [baseline mesh] [water vapor var] ' \
            '[first step] [last step]\n\n')
        sys.exit(-1)

    # parse command line
    deeplab_model = sys.argv[1]
    resnet_model = sys.argv[2]
    mesh_data_regex = sys.argv[3]
    baseline_mesh = sys.argv[4]
    water_vapor_var = sys.argv[5]
    first_step =  int(sys.argv[6])
    last_step = int(sys.argv[7])

    cf_reader = teca_cf_reader.New()
    cf_reader.set_files_regex(mesh_data_regex)
    cf_reader.set_periodic_in_x(1)

    # Getting padding parameters based on the lon & lat sizes
    md = cf_reader.update_metadata()

    try:
        atrs = md["attributes"]
    except:
        raise KeyError("metadata missing attributes")

    try:
        lat = atrs['lat']['dims']
        lon = atrs['lon']['dims']
    except:
        raise KeyError(
                "failed to determine the geographic coordinates")

    # Setting the padding dimensions
    padding_amount_lat, padding_amount_lon = get_padding_params(lat, lon)

    # Continue original pipeline
    mesh_padder = teca_mesh_padding.New()
    mesh_padder.set_input_connection(cf_reader.get_output_port())
    mesh_padder.set_py_low(padding_amount_lat[0])
    mesh_padder.set_py_high(padding_amount_lat[1])
    mesh_padder.set_px_low(padding_amount_lon[0])
    mesh_padder.set_px_high(padding_amount_lon[1])

    deeplabv3p_ar_detect = teca_deeplabv3p_ar_detect.New()
    deeplabv3p_ar_detect.set_input_connection(mesh_padder
        .get_output_port())
    deeplabv3p_ar_detect.set_variable_name(water_vapor_var)

    deeplabv3p_ar_detect.build_model(
        deeplab_model,
        resnet_model
    )

    if os.path.exists(baseline_mesh):
        # run the test
        baseline_mesh_reader = teca_cartesian_mesh_reader.New()
        baseline_mesh_reader.set_file_name(baseline_mesh)

        diff = teca_dataset_diff.New()
        diff.set_input_connection(0, baseline_mesh_reader.get_output_port())
        diff.set_input_connection(1, deeplabv3p_ar_detect.get_output_port())
        diff.update()
    else:
        # make a baseline
        if rank == 0:
            print("generating baseline image", baseline_mesh)
        wri = teca_cartesian_mesh_writer.New()
        wri.set_input_connection(deeplabv3p_ar_detect.get_output_port())
        wri.set_file_name(baseline_mesh)

        wri.update()


if __name__ == '__main__':
    main()