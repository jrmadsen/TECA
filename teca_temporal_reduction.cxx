#include "teca_temporal_reduction.h"
#include "teca_binary_stream.h"

#if defined(TECA_MPI)
#include <mpi.h>
#endif

using std::vector;
using std::cerr;
using std::endl;

// TODO
// handle large messages
namespace {
#if defined(TECA_MPI)
// helper for sending binary data over MPI
int send(MPI_Comm comm, int dest, teca_binary_stream &s)
{
    unsigned long long n = s.size();
    if (MPI_Send(&n, 1, MPI_UNSIGNED_LONG_LONG, dest, 3210, comm))
    {
        TECA_ERROR("failed to send send message size")
        return -1;
    }

    if (MPI_Send(
            s.get_data(),
            n,
            MPI_UNSIGNED_CHAR,
            dest,
            3211,
            MPI_COMM_WORLD))
    {
        TECA_ERROR("failed to send message")
        return -2;
    }

    return 0;
}

// helper for receiving data over MPI
int recv(MPI_Comm comm, int src, teca_binary_stream &s)
{
    size_t n = 0;
    MPI_Status stat;
    if (MPI_Recv(&n, 1, MPI_UNSIGNED_LONG_LONG, src, 3210, comm, &stat))
    {
        TECA_ERROR("failed to receive")
        return -2;
    }

    s.resize(n);

    if (MPI_Recv(s.get_data(), n, MPI_UNSIGNED_CHAR, src, 3211, comm, &stat))
    {
        TECA_ERROR("failed to receive")
        return -2;
    }

    return 0;
}
#endif
};

// --------------------------------------------------------------------------
std::vector<teca_meta_data> teca_temporal_reduction::get_upstream_request(
    unsigned int port,
    const std::vector<teca_meta_data> &input_md,
    const teca_meta_data &request)
{
    vector<teca_meta_data> up_req;

    // locate available times
    vector<double> time;
    if (input_md[0].get_prop("time", time))
    {
        TECA_ERROR("missing time metadata")
        return up_req;
    }

    // partition time across MPI ranks. each rank
    // will end up with a unique block of times
    // to process.
    size_t rank = 0;
    size_t n_ranks = 1;
#if defined(TECA_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        int tmp = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &tmp);
        n_ranks = tmp;
        MPI_Comm_rank(MPI_COMM_WORLD, &tmp);
        rank = tmp;
    }
#endif
    size_t n_times = time.size();
    size_t n_big_blocks = n_times%n_ranks;
    size_t block_size = 1;
    size_t block_start = 0;
    if (rank < n_big_blocks)
    {
        block_size = n_times/n_ranks + 1;
        block_start = block_size*rank;
    }
    else
    {
        block_size = n_times/n_ranks;
        block_start = block_size*rank + n_big_blocks;
    }

    // get the filters basic request
    vector<teca_meta_data> base_req
        = this->initialize_upstream_request(port, input_md, request);

    // apply the base request to local times.
    // requests are mapped onto inputs round robbin
    for (size_t i = 0; i < block_size; ++i)
    {
        size_t ii = i + block_start;
        size_t n_reqs = base_req.size();
        for (size_t j = 0; j < n_reqs; ++j)
        {
            up_req.push_back(base_req[j]);
            up_req.back().set_prop("time", time[ii]);
        }
    }

    return up_req;
}

// --------------------------------------------------------------------------
teca_meta_data teca_temporal_reduction::get_output_meta_data(
    unsigned int port,
    const std::vector<teca_meta_data> &input_md)
{
    teca_meta_data output_md
        = this->initialize_output_meta_data(port, input_md);

    output_md.remove_prop("time");

    return output_md;
}

// --------------------------------------------------------------------------
p_teca_dataset teca_temporal_reduction::reduce_local(
    std::vector<p_teca_dataset> input_data)
{
    size_t n_in = input_data.size();

    if (n_in == 0)
        return p_teca_dataset();

    if (n_in == 1)
        return input_data[0];

    while (n_in > 1)
    {
        if (n_in % 2)
            input_data[0] = this->reduce(input_data[0], input_data[n_in-1]);

        size_t n = n_in/2;
        for (size_t i = 0; i < n; ++i)
        {
            size_t ii = 2*i;
            input_data[i] = this->reduce(input_data[ii], input_data[ii+1]);
        }

        n_in = n;
    }
    return input_data[0];
}

// --------------------------------------------------------------------------
p_teca_dataset teca_temporal_reduction::reduce_remote(
    p_teca_dataset local_data)
{
#if defined(TECA_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        size_t rank = 0;
        size_t n_ranks = 1;
        int tmp = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &tmp);
        n_ranks = tmp;
        MPI_Comm_rank(MPI_COMM_WORLD, &tmp);
        rank = tmp;

        // special case 1 rank, nothing to do
        if (n_ranks < 2)
            return local_data;

        // reduce remote datasets in binary tree order
        size_t id = rank + 1;
        size_t up_id = id/2;
        size_t left_id = 2*id;
        size_t right_id = left_id + 1;

        teca_binary_stream bstr;

        // recv from left
        if (left_id <= n_ranks)
        {
            if (::recv(MPI_COMM_WORLD, left_id-1, bstr))
            {
                TECA_ERROR("failed to recv from left")
                return p_teca_dataset();
            }
            p_teca_dataset left_data = local_data->new_instance();
            left_data->from_stream(bstr);
            local_data = this->reduce(local_data, left_data);

            bstr.resize(0);
        }

        // recv from right
        if (right_id <= n_ranks)
        {
            if (::recv(MPI_COMM_WORLD, right_id-1,  bstr))
            {
                TECA_ERROR("failed to recv from right")
                return p_teca_dataset();
            }
            p_teca_dataset right_data = local_data->new_instance();
            right_data->from_stream(bstr);
            local_data = this->reduce(local_data, right_data);

            bstr.resize(0);
        }

        // send up
        if (rank)
        {
            local_data->to_stream(bstr);

            if (::send(MPI_COMM_WORLD, up_id-1, bstr))
                TECA_ERROR("failed to send up")

            // all but root returns an empty dataset
            return p_teca_dataset();
        }
    }
#endif
    // rank 0 has all the data
    return local_data;
}

// --------------------------------------------------------------------------
p_teca_dataset teca_temporal_reduction::execute(
    unsigned int port,
    const std::vector<p_teca_dataset> &input_data,
    const teca_meta_data &request)
{
    size_t n_in = input_data.size();
    if (n_in == 0)
    {
        TECA_ERROR("empty input")
        return p_teca_dataset();
    }

    return this->reduce_remote(this->reduce_local(input_data));
}
