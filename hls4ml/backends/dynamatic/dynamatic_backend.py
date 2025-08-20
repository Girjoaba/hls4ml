# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Any, TYPE_CHECKING
from numpy.typing import NDArray
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer
    from subprocess import CompletedProcess

import os, sys
import re
import subprocess, shlex
import numpy as np
from warnings import warn
from fxpmath import Fxp

from hls4ml.backends import FPGABackend
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.flow import register_flow
from hls4ml.model.attributes import ChoiceAttribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.layers import (
    Dense,
    Layer,
    Activation,
    Softmax
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

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

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
            + templates
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
            template_flow,
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

        return config


    def compile(self, model: ModelGraph) -> None:
        raise Exception('COMPILE not implemented yet...')
        path = self._get_backend_exec_path(model)

        curr_dir = os.getcwd()
        os.chdir(f'{model.config.get_output_dir()}/firmware')
        kernel_name = model.config.get_project_name()

        ## Generate IR
        with open(f'{kernel_name}.ir', 'w') as ir_file:
            gen_cmd = [ 
                f'{path}/xls/dslx/ir_convert/ir_converter_main',
                f'--top={kernel_name}',
                f'{kernel_name}.x'
            ]
            subprocess.run(gen_cmd, check=True, stdout=ir_file)
        ## Optimize IR
        with open(f'{kernel_name}.opt.ir', 'w') as opt_file:
            opt_cmd = [ 
                f'{path}/xls/tools/opt_main',
                f'{kernel_name}.ir'
            ]
            subprocess.run(opt_cmd, check=True, stdout=opt_file)

        os.chdir(curr_dir)

    def build(
        self,
        model,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
    ):
        raise Exception('BUILD not implemented yet...')
        if 'linux' in sys.platform:
            found = os.system('command -v vivado_hls > /dev/null')
            if found != 0:
                raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')

        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        vivado_cmd = (
            f'vivado_hls -f build_prj.tcl "reset={reset} '
            f'csim={csim} '
            f'synth={synth} '
            f'cosim={cosim} '
            f'validation={validation} '
            f'export={export} '
            f'vsynth={vsynth} '
            f'fifo_opt={fifo_opt}"'
        )
        os.system(vivado_cmd)
        os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())

   