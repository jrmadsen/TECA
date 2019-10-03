#include "teca_mesh_padding.h"
#include "teca_mesh_layering.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_dataset_capture.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_dataset_source.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>

using namespace std;


int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    cerr << "argc: " << argc << endl;
    if (argc != 8)
    {
        cerr << "Usage: test_mesh_pad_layer [nx] [ny] [px_low] [px_high] " <<
         "[py_low] [py_high] [out file]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    unsigned long px_low = atoi(argv[3]);
    unsigned long px_high = atoi(argv[4]);
    unsigned long py_low = atoi(argv[5]);
    unsigned long py_high = atoi(argv[6]);
    string out_file = argv[7];

    // allocate a mesh
    // coordinate axes
    using coord_t = double;
    coord_t dx = coord_t(360.)/coord_t(nx - 1);
    p_teca_variant_array_impl<coord_t> x = teca_variant_array_impl<coord_t>::New(nx);
    coord_t *px = x->get();
    for (unsigned long i = 0; i < nx; ++i)
        px[i] = i*dx;

    coord_t dy = coord_t(180.)/coord_t(ny - 1);
    p_teca_variant_array_impl<coord_t> y = teca_variant_array_impl<coord_t>::New(ny);
    coord_t *py = y->get();
    for (unsigned long i = 0; i < ny; ++i)
        py[i] = coord_t(-90.) + i*dy;

    unsigned long nz = 1;
    p_teca_variant_array_impl<coord_t> z = teca_variant_array_impl<coord_t>::New(nz);
    z->set(0, 0.f);

    p_teca_variant_array_impl<coord_t> t = teca_variant_array_impl<coord_t>::New(1);
    t->set(0, 1.f);

    unsigned long nxy = nx * ny;
    p_teca_double_array ones_grid = teca_double_array::New(nxy);
    double *p_ones_grid = ones_grid->get();

    for (unsigned int i = 0; i < nxy; ++i)
    {
        p_ones_grid[i] = 1;
    }

    unsigned long wext[] = {0, nx - 1, 0, ny - 1, 0, 0};

    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates("x", x);
    mesh->set_y_coordinates("y", y);
    mesh->set_z_coordinates("z", z);
    mesh->set_whole_extent(wext);
    mesh->set_extent(wext);
    mesh->set_time(1.0);
    mesh->set_time_step(0ul);
    mesh->get_point_arrays()->append("ones_grid", ones_grid);

    teca_metadata md;
    md.set("whole_extent", wext, 6);
    md.set("time_steps", std::vector<unsigned long>({0}));
    md.set("variables", std::vector<std::string>({"ones_grid"}));
    md.set("number_of_time_steps", 1);
    md.set("index_initializer_key", std::string("number_of_time_steps"));
    md.set("index_request_key", std::string("time_step"));

    // build the pipeline
    p_teca_dataset_source source = teca_dataset_source::New();
    source->set_metadata(md);
    source->set_dataset(mesh);

    p_teca_mesh_padding mesh_padder = teca_mesh_padding::New();
    mesh_padder->set_input_connection(source->get_output_port());
    mesh_padder->set_px_low(px_low);
    mesh_padder->set_px_high(px_high);
    mesh_padder->set_py_low(py_low);
    mesh_padder->set_py_high(py_high);

    p_teca_mesh_layering mesh_layerer = teca_mesh_layering::New();
    mesh_layerer->set_input_connection(mesh_padder->get_output_port());
    mesh_layerer->set_n_layers(3);

    //p_teca_dataset_capture transformed_o = teca_dataset_capture::New();
    //transformed_o->set_input_connection(mesh_layerer->get_output_port());
    //transformed_o->set_input_connection(mesh_padder->get_output_port());

    p_teca_index_executive exe = teca_index_executive::New();
    exe->set_start_index(0);
    exe->set_end_index(0);

    p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
    wri->set_input_connection(mesh_layerer->get_output_port());
    wri->set_executive(exe);
    wri->set_file_name(out_file);

    wri->update();

    /*
    const_p_teca_dataset ds = transformed_o->get_dataset();
    const_p_teca_cartesian_mesh cds = std::dynamic_pointer_cast<const teca_cartesian_mesh>(ds);
    const_p_teca_variant_array va = cds->get_point_arrays()->get("ones_grid");

    using TT = teca_variant_array_impl<double>;
    using NT = double;

    const NT *p_transformed_array = dynamic_cast<const TT*>(va.get())->get();

    for (long k = 0; k < long(nz); ++k)
    {
        for (long j = 0; j < long(ny); ++j)
        {
            for (long i = 0; i < long(nx); ++i)
            {
                cerr << p_transformed_array[k*ny*nx + j*nx + i];
            }
            cerr << endl;
        }
        cerr << endl << endl;
    }
    */

    return 0;
}
