#ifndef teca_latitude_damper_h
#define teca_latitude_damper_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_latitude_damper)

/**
damps the specified scalar field(s) using an inverted Gaussian centered on a
given latitude with a half width specified in degrees latitude. The paramters
defining the Gaussian (center, half width at half max) can be specified by the
user directly or by down stream algorithm via the following keys in the request.

request keys:

  teca_latitude_damper::damped_variables
  teca_latitude_damper::half_width_at_half_max
  teca_latitude_damper::center

note that user specified values take precedence over request keys. When using
request keys be sure to include the variable post-fix.
*/
class teca_latitude_damper : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_latitude_damper)
    ~teca_latitude_damper();

    // TODO -- hanlde command line arguments

    // set the center of the Gaussian in units of degress latitude.
    // default is 0.0 deg lat
    TECA_ALGORITHM_PROPERTY(double, center)

    // set the half width of the Gaussian in units of degrees latitude.
    // default is 45.0 deg lat
    TECA_ALGORITHM_PROPERTY(double, half_width_at_half_max)

    // set the names of the arrays that the filter will apply on
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, damped_variable)

    // a string to be appended to the name of each output variable
    // setting this to an empty string will result in the damped array
    // replacing the input array in the output. default is an empty
    // string ""
    TECA_ALGORITHM_PROPERTY(std::string, variable_post_fix)

protected:
    teca_latitude_damper();

    // helpers to get parameters defining the Gaussian used by the
    // the filter. if the user has not specified a value then the
    // request is probed. a return of zero indicates success
    int get_sigma(const teca_metadata &request, double &sigma);
    int get_mu(const teca_metadata &request, double &mu);

    // helper to get the list of variables to apply the filter on
    // if the user provided none, then the request is probed. a
    // return of 0 indicates success
    int get_damped_variables(const teca_metadata &request,
        std::vector<std::string> &vars);

private:
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    double center;
    double half_width_at_half_max;
    std::vector<std::string> damped_variables;
    std::string variable_post_fix;
};

#endif