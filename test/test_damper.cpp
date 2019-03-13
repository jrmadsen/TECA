#include "teca_cf_reader.h"
#include "teca_damper.h"
#include "teca_l2_norm.h"
#include "teca_connected_components.h"
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_time_step_executive.h"
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


bool isequal(double a, double b, double epsilon)
{
    return fabs(a - b) < epsilon;
}

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    // ======================================================
    if (argc != 5)
    {
        cerr << "test_damper [nx] [ny] [hwhm] [out file]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    double hwhm = double(atoi(argv[3]));
    string out_file = argv[4];

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

    p_teca_variant_array_impl<coord_t> z = teca_variant_array_impl<coord_t>::New(1);
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

    // Calculating sigma = hwhm/sqrt(2*ln(2))
    double sigma = hwhm / 1.17741;
    //cerr << "sigma: " << sigma << endl;

    unsigned long wext[] = {0, nx - 1, 0, ny - 1, 0, 0};

    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates(x);
    mesh->set_y_coordinates(y);
    mesh->set_z_coordinates(z);
    mesh->set_whole_extent(wext);
    mesh->set_extent(wext);
    mesh->set_time(1.0);
    mesh->set_time_step(0ul);
    mesh->get_point_arrays()->append("ones_grid", ones_grid);

    teca_metadata md;
    md.insert("whole_extent", wext, 6);
    md.insert("time_steps", std::vector<unsigned long>({0}));
    md.insert("variables", std::vector<std::string>({"ones_grid"}));
    md.insert("number_of_time_steps", 1);

    // build the pipeline
    p_teca_dataset_source source = teca_dataset_source::New();
    source->set_metadata(md);
    source->set_dataset(mesh);

    p_teca_damper damped_comp = teca_damper::New();
    damped_comp->set_filter_lat_width_value(sigma);
    damped_comp->append_damper_variable("ones_grid");
    damped_comp->set_input_connection(source->get_output_port());

    p_teca_dataset_capture damp_o = teca_dataset_capture::New();
    damp_o->set_input_connection(damped_comp->get_output_port());

    p_teca_time_step_executive exe = teca_time_step_executive::New();
    exe->set_first_step(0);
    exe->set_last_step(0);

    p_teca_vtk_cartesian_mesh_writer wri = teca_vtk_cartesian_mesh_writer::New();
    wri->set_input_connection(damp_o->get_output_port());
    wri->set_executive(exe);
    wri->set_file_name(out_file);

    wri->update();

    
    const_p_teca_dataset ds = damp_o->get_dataset();
    const_p_teca_cartesian_mesh cds = std::dynamic_pointer_cast<const teca_cartesian_mesh>(ds);
    const_p_teca_variant_array va = cds->get_point_arrays()->get("ones_grid_damped");
    
    using TT = teca_variant_array_impl<double>;
    using NT = double;

    const NT *p_damped_array = static_cast<const TT*>(va.get())->get();
    int hwhm_index = -1;

    for (unsigned long j = 0; j < ny; ++j)
    {
        if (isequal(py[j], hwhm, 1e-7))
        {
            hwhm_index = (int) (j * nx);
            break;
        }
    }

    cerr << "hwhm_index: " << hwhm_index << endl;
    if (hwhm_index > 0 && 
        isequal(p_damped_array[hwhm_index], 0.5, 1e-7))
    {
        cerr << "p_damped_array[hwhm_index]: " << p_damped_array[hwhm_index] << endl;
    }
    else
    {
        TECA_ERROR("Damping failed!")
        return -1;
    }

    return 0;
}