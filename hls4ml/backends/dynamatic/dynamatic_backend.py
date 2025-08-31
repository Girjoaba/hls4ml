# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import Any, TYPE_CHECKING
from numpy.typing import NDArray
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer
    from subprocess import CompletedProcess

import os, glob, re, sys
import subprocess
import numpy as np
from warnings import warn
from fxpmath import Fxp

from hls4ml.backends import FPGABackend
from hls4ml.model.optimizer import get_backend_passes
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import (
    Layer,
)


class DynamaticBackend(FPGABackend):
    def __init__(self):
        super().__init__('Dynamatic')
        self._register_flows()


    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        optimization_passes = [
            'infer_precision_types',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        dynamatic_attributes = [
            'dynamatic:build_attr',
        ]
        dynamatic_attributes_flow: str = register_flow('specific_attributes', dynamatic_attributes, requires=[optimization_flow], backend=self.name)

        dynamatic_optimization_passes = [
            'dynamatic:merge_dense_relu',
        ]
        dynamatic_optimization_passes_flow: str = register_flow('merge_dense_relu_layers', dynamatic_optimization_passes, requires=[dynamatic_attributes_flow], backend=self.name)

        writer_passes = ['dynamatic:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['dynamatic:ip'], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + optimization_passes
            + writer_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            optimization_flow,
            dynamatic_attributes_flow,
            dynamatic_optimization_passes_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(
        self,
        part='xcvu13p-flga2577-2-e',
        clock_period=5,
        clock_uncertainty='12.5%',
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        tb_output_stream='both',
        **_,
    ):
        """Create initial configuration of the Vivado backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'xcvu13p-flga2577-2-e'.
            clock_period (int, optional): The clock period. Defaults to 5.
            clock_uncertainty (str, optional): The clock uncertainty. Defaults to 12.5%.
            io_type (str, optional): Type of implementation used. One of
                'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
            namespace (str, optional): If defined, place all generated code within a namespace. Defaults to None.
            write_weights_txt (bool, optional): If True, writes weights to .txt files which speeds up compilation.
                Defaults to True.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.
            tb_output_stream (str, optional): Controls where to write the output. Options are 'stdout', 'file' and 'both'.
                Defaults to 'both'.

        Returns:
            dict: initial configuration.
        """
        config = {}

        config['Part'] = part if part is not None else 'xcvu13p-flga2577-2-e'
        config['ClockPeriod'] = clock_period if clock_period is not None else 5
        config['ClockUncertainty'] = clock_uncertainty if clock_uncertainty is not None else '12.5%'
        config['IOType'] = io_type if io_type is not None else 'io_parallel'
        config['HLSConfig'] = {}
        config['WriterConfig'] = {
            'Namespace': namespace,
            'WriteWeightsTxt': write_weights_txt,
            'WriteTar': write_tar,
            'TBOutputStream': tb_output_stream,
        }
        #TODO: update to a better way to access the project
        config['dynamatic_path'] = '$HOME/dynamatic'

        return config

    def _get_backend_exec_path(self, model: ModelGraph) -> str:
        if 'linux' in sys.platform:
            path: str = os.path.expandvars(model.config.get_config_value('dynamatic_path'))
            if not os.path.isdir(path):
                raise Exception('Dynamatic is expected to be installed in your $HOME dir. We are looking for `$HOME/dynamatic`')
        return path

    def compile(self, model: ModelGraph) -> None:
        """Compiles the Dynamatic project by calling Dynamatic to generate HDL from the written C-code
        and creating an executable called for model prediction.

        Args:
            model (ModelGraph): hls4ml IR containining the neural network information.
        """
        path = self._get_backend_exec_path(model)

        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        ## Run Dynamatic
        gen_cmd = [ 
            f'bash',
            f'../llvm-cf-handshake.sh',
            f'{path}',
            f'{kernel_name}.c',
            f'{kernel_name}'
        ]
        subprocess.run(gen_cmd, check=True)

        os.chdir(curr_dir)


    def predict(self, model: ModelGraph, x: np.floating | NDArray[np.floating[Any]]) -> list[NDArray[np.floating]]:
        """Takes multiple samples and runs them through the simulated network. 
        All the predictions are added to a vector and returned.

        Args:
            model (ModelGraph): hls4ml IR containining the neural network information.
            x (numpy array): input vectors
        Returns:
            numpy array: network predictions.

        TODO: make the prediction executable a linked .so file
        """
        def _read_output(output_width: int, kernel_name: str) -> list:
            """Read the output and convert it to an integer."""

            # find output files
            folder = f"./out-{kernel_name}/sim/C_OUT"
            matches = sorted(
                glob.glob(os.path.join(folder, "output_out*_*.dat")),
                key=lambda f: int(re.search(r"_(\d+)\.dat$", os.path.basename(f)).group(1))
            )
            if not matches:
                raise FileNotFoundError(f"No files matched {folder}/output_out*.dat")

            # read outputs files
            ints = []
            for filepath in matches:
                inside_tx = False
                hex_pattern = re.compile(r"0x[0-9a-fA-F]+")

                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("[[transaction]]"):
                            inside_tx = True
                            continue
                        if line.startswith("[[/transaction]]"):
                            inside_tx = False
                            continue
                        if inside_tx:
                            for tok in hex_pattern.findall(line):
                                ints.append(int(tok, 16))

            # signed interpretation w/ 2's complement to obtain the de-quantized integer value
            sign_bit = 1 << (output_width - 1)
            full_mask = 1 << output_width
            sint_output = [(v - full_mask) if (v & sign_bit) else v for v in ints]

            return [sint_output]

        def _predict_using_c_model(model: ModelGraph, 
                             path: str, 
                             x_list: NDArray[np.floating], 
                             n_samples: int, 
                             n_inputs: int, 
                             input_width: int, 
                             input_frac: int) -> list:
            kernel_name = model.config.get_project_name()
            predict_cmd = [ 
                f'bash',
                f'../predict.sh',
                f'{path}',
                f'{kernel_name}.c',
                f'{kernel_name}'
            ]
            results = []

            for i in range(n_samples):
                # format the input to be fed into the C model as a text file
                if n_inputs == 1:
                    inp = [np.asarray(x_list[i])]
                else:
                    inp = [np.asarray(xj) for xj in x_list[i]]
                fxp_x: list[NDArray[np.int_]] = Fxp(inp, signed=True, n_word=input_width, n_frac=input_frac).raw() 
                newline = ''
                if n_inputs == 1:
                    newline += f'{fxp_x[0][0]}'
                else:
                    for i, inp in enumerate(fxp_x):
                        newline += f'{inp} '
                with open('input.txt', 'w') as f:
                    f.write(newline)
                # run command
                subprocess.run(predict_cmd, check=True)
                output = _read_output(input_width, kernel_name)
                results += output

            return results


        def _go_to_original_type(rows: list, 
                                 python_input_type: np.dtype[np.floating], 
                                 scale) -> list[NDArray[np.floating]]:
            output = np.array(rows, dtype=np.int32).astype(python_input_type) / scale
            return output

        path: str = self._get_backend_exec_path(model)
        layers: list[Layer] = list(model.get_layers())

        # Extract dimensions
        n_samples : int = model._compute_n_samples(x)
        n_inputs  : int = list(layers[0].get_output_variable().get_shape())[0][1] # Get input dimensions

        # Extract type
        input_width : int = list(layers[0].get_layer_precision().items())[0][1].precision.width
        input_frac  : int = input_width - list(layers[0].get_layer_precision().items())[0][1].precision.integer
        output_width: int = list(layers[len(layers)-1].get_layer_precision().items())[0][1].precision.width
        output_frac : int = output_width - list(layers[len(layers)-1].get_layer_precision().items())[0][1].precision.integer

        # extract python type (float/double)
        if isinstance(x, np.ndarray):
            python_input_type: np.dtype[np.floating] = x[0].dtype
        else:
            python_input_type: np.dtype[np.floating]  = x.dtype
        
        if n_samples == 1 and n_inputs == 1 and isinstance(x, np.floating):
            x_list: NDArray[np.floating] = np.array([x], dtype=x.dtype)
        elif isinstance(x, np.ndarray): 
            x_list: NDArray[np.floating] = x

        # Change dirs
        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')

        # Result processing pipeling
        result = _predict_using_c_model(model, path, x_list, n_samples, n_inputs, input_width, input_frac)
        os.chdir(curr_dir)
        result_floats: list[NDArray[np.floating]] = _go_to_original_type(result, python_input_type, scale=2 ** output_frac)
        return result_floats

    def build(
        self,
        model: ModelGraph,
        full_clock: float = 5
    ):
        """Generates the synthesized design with timing and resource reports.

        Args:
            model (ModelGraph): hls4ml IR containining the neural network information.
            full_clock (float): clock period in nanoseconds (ns)
        """
        path = self._get_backend_exec_path(model)

        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        ## Run Dynamatic
        gen_cmd = [ 
            f'bash',
            f'../synthesize.sh',
            f'{path}',
            f"./out-{kernel_name}",
            f'{kernel_name}',
            f'{full_clock}',
            f'{full_clock/2}'
        ]
        subprocess.run(gen_cmd, check=True)

        os.chdir(curr_dir)

        # TODO: implement parse_report
        # return parse_vivado_report(model.config.get_output_dir())

   