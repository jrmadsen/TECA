import sys
import teca_py
import numpy as np
import torch
import torch.nn.functional as F
import time


class teca_model_segmentation(teca_py.teca_python_algorithm):
    """
    Given an input field of integrated vapor transport,
    calculates the probability of AR presence at each gridcell.
    """
    def __init__(self):
        self.variable_name = "IVT"
        self.pred_name = self.variable_name + "_PRED"
        self.var_array = None
        self.transform_fn = None
        self.transport_fn_args = None
        self.model = None
        self.model_path = None
        self.device = self.set_torch_device()

    def __str__(self):
        ms_str = 'variable_name=%s, pred_name=%d\n\n'%( \
            self.variable_name, self.pred_name)

        ms_str += 'model:\n%s\n\n'%(str(self.model))

        ms_str += 'device:\n%s\n'%(str(self.device))

        return ms_str

    def load_state_dict(self, state_dict_file):
        """
        Load only the pytorch state_dict parameters file only
        once and broadcast it to all ranks
        """
        comm = self.get_communicator()
        rank = comm.Get_rank()

        sd = None
        if rank == 0:
            sd = torch.load(state_dict_file, map_location=lambda storage, loc: storage)
        sd = comm.bcast(sd, root=0)
        #sd = torch.load(state_dict_file, map_location=lambda storage, loc: storage)
        
        return sd

    def set_variable_name(self, name):
        """
        set the variable name that will be inputed to the model
        """
        self.variable_name = str(name)
        self.set_pred_name(self.variable_name + "_pred")

    def set_pred_name(self, name):
        """
        set the variable name that will be the output to the model
        """
        self.pred_name = name

    def __set_transform_fn(self, fn, *args):
        """
        if the data need to be transformed in a way then a function
        could be provided to be applied on the requested data before
        running it to the model.
        """
        if not hasattr(fn, '__call__'):
            if rank == 0:
                raise TypeError("ERROR: The provided data transform function "
                    "is not a function")

        if not args:
            if rank == 0:
                raise ValueError("ERROR: The provided data transform function "
                    "must at least have 1 argument -- the data array object to "
                    "apply the transformation on.")

        self.transform_fn = fn
        self.transport_fn_args = args

    def set_torch_device(self, device="cpu"):
        """
        Set device to either 'cuda' or 'cpu'
        """
        if device == "cuda" and not torch.cuda.is_available():
            if rank == 0:
                raise Exception("ERROR: Couldn\'t set device to cuda, cuda is "
                    "not available")

        return torch.device(device)

    def set_model(self, model):
        """
        set Pytorch pretrained model
        """
        self.model = model
        self.model.eval()

    def get_report_callback(self):
        """
        return a teca_algorithm::report function adding the output name
        that will hold the output predictions of the used model.
        """
        def report(port, rep_in):
            rep_temp = rep_in[0]

            rep = teca_py.teca_metadata(rep_temp)

            if not rep.has('variables'):
                print("variables key doesn't exist")
                rep['variables'] = teca_py.teca_variant_array.New(np.array([]))

            if self.pred_name:
                rep.append("variables", self.pred_name)

            return rep
        return report

    def get_request_callback(self):
        """
        return a teca_algorithm::request function adding the variable name
        that the pretrained model will process.
        """
        def request(port, md_in, req_in):
            if not self.variable_name:
                if rank == 0:
                    raise ValueError("ERROR: No variable to request specifed")

            req = teca_py.teca_metadata(req_in)

            arrays = []
            if req.has('arrays'):
                arrays = req['arrays']

            arrays.append(self.variable_name)
            req['arrays'] = arrays

            return [req]
        return request

    def get_execute_callback(self):
        """
        return a teca_algorithm::execute function
        """
        def execute(port, data_in, req):
            """
            expects an array of an input variable to run through
            the torch model and get the segmentation results as an
            output.
            """
            in_mesh = teca_py.as_teca_cartesian_mesh(data_in[0])

            if in_mesh is None:
                if rank == 0:
                    raise ValueError("ERROR: empty input, or not a mesh")

            if self.model is None:
                if rank == 0:
                    raise ValueError("ERROR: pretrained model has not been specified")

            #md = in_mesh.get_metadata()

            #arrays = in_mesh.get_point_arrays()

            #var_array = arrays[self.variable_name]
            
            #if self.transform_fn:
            #    var_array = self.transform_fn(var_array, *self.transport_fn_args)

            start_time = time.time()
            self.var_array = torch.from_numpy(self.var_array).to(self.device)
            end_time = time.time()

            #print(self.var_array)
            start_time = time.time()
            with torch.no_grad():
                pred = torch.sigmoid(self.model(self.var_array))
            end_time = time.time()

            if pred is None:
                if rank == 0:
                    raise Exception("ERROR: Model failed to get predictions")

            out_mesh = teca_py.teca_cartesian_mesh.New()
            out_mesh.shallow_copy(in_mesh)

            pred = teca_py.teca_variant_array.New(pred.numpy().ravel())
            out_mesh.get_point_arrays().set(self.pred_name, pred)

            return out_mesh
        return execute
