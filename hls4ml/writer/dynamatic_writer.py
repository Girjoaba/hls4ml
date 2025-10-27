# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph

import os
from shutil import copyfile, copytree, rmtree
from hls4ml.writer.writers import Writer


config_filename = 'hls4ml_config.yml'


class DynamaticWriter(Writer):
    
    def write_project_dir(self, model: ModelGraph) -> None:
        """Write the base project directory

        Args:
            model (ModelGraph): hls4ml IR containining the neural network information.
        """
        if not os.path.isdir(f"{model.config.get_output_dir()}/firmware"):
            os.makedirs(f"{model.config.get_output_dir()}/firmware")

        input_txt_path = os.path.join(model.config.get_output_dir(), "firmware", "input.txt")
        with open(input_txt_path, "w") as f_input:
            f_input.write("")


    def write_scripts(self, model: ModelGraph) -> None:
        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath_compile = os.path.join(filedir, '../templates/dynamatic/llvm-cf-handshake.sh')
        dstpath_compile = f'{model.config.get_output_dir()}/llvm-cf-handshake.sh'
        copyfile(srcpath_compile, dstpath_compile)

        srcpath_predict = os.path.join(filedir, '../templates/dynamatic/predict.sh')
        dstpath_predict = f'{model.config.get_output_dir()}/predict.sh'
        copyfile(srcpath_predict, dstpath_predict)

        srcpath_synth = os.path.join(filedir, '../templates/dynamatic/synthesize.sh')
        dstpath_synth = f'{model.config.get_output_dir()}/synthesize.sh'
        copyfile(srcpath_synth, dstpath_synth)


    def write_project_dynamatic(self, model: ModelGraph) -> None:
        """Write the main architecture source file (myproject.c)

        Args:
            model (ModelGraph): hls4ml IR containining the neural network information.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/dynamatic/firmware/myproject.c'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.c', 'w')

        layers = list(model.get_layers())
        indent = '    '
        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert dimensions' in line:
                newline = line
                for i, layer in enumerate(layers):
                    if layer.get_attr("write_dims"):
                        newline += f'#define {layer.get_attr("out_dim_key")} {layer.get_attr("out_dim_val")}\n'

            elif '// hls-fpga-machine-learning architecture arguments' in line:
                newline = ''
                func_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        for input_idx in range(layer.get_attr("out_dim_val")):
                            newline += indent + f'default_t input_{input_idx}, \n'
                    elif layer.get_attr('write_func'):
                        if func_layers_count == len([layer for layer in layers if layer.get_attr("write_func")]) - 1:
                            for output_idx in range(layer.get_attr("out_dim_val")):
                                newline += indent + f'default_t out{i}_{output_idx}[1]'
                                if output_idx < layer.get_attr("out_dim_val") - 1:
                                    newline += ','
                                newline += '\n'
                        func_layers_count += 1

            elif '// hls-fpga-machine-learning intermediate stores' in line:
                newline = line
                func_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        newline += indent + f'default_t tmp_input[{layer.get_attr("out_dim_key")}];\n'
                        for input_idx in range(layer.get_attr("out_dim_val")):
                            newline += indent + f'tmp_input[{input_idx}] = input_{input_idx};\n'
                    elif layer.get_attr('write_func'):
                        if func_layers_count < len([layer for layer in layers if layer.get_attr("write_func")]):
                            newline += indent + f'default_t out{i}[{layer.get_attr("out_dim_key")}];\n'
                            func_layers_count += 1


            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line
                prev_var = 'tmp_input'
                for i, layer in enumerate(layers):
                    if layer.get_attr('write_func'):
                        if layer.get_attr('write_weights'):
                            newline += indent + f'dense_accum_t acc{i};\n'
                            newline += indent + f'default_t tmp{i};\n'
                            newline += indent + f'{layer.get_attr("func_call")}({prev_var}, out{i}, {layer.get_attr("in_dim_key")}, {layer.get_attr("out_dim_key")}, w{i}, b{i}, acc{i}, tmp{i});\n'
                            prev_var = f'out{i}'
                        else:
                            newline += indent + f'default_t tmp{i};\n'
                            newline += indent + f'{layer.get_attr("func_call")}({prev_var}, out{i}, {layer.get_attr("in_dim_key")}, tmp{i});\n'
                            prev_var = f'out{i}'                            

            elif '// hls-fpga-machine-learning write outputs' in line:
                newline = line
                func_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.get_attr('write_func'):
                        if func_layers_count == len([layer for layer in layers if layer.get_attr("write_func")]) - 1:
                            for output_idx in range(layer.get_attr("out_dim_val")):
                                newline += indent + f'out{i}_{output_idx}[0] = out{i}[{output_idx}];\n'
                        func_layers_count += 1

            elif '// hls-fpga-machine-learning input init' in line:
                newline = line
                func_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        newline += indent + f'default_t input[{layer.get_attr("out_dim_key")}];\n'
                        newline += indent + f'FILE *f = fopen("input.txt", "r");\n'
                        for input_idx in range(layer.get_attr("out_dim_val")):
                            newline += indent + f'default_t input_{input_idx};\n'
                            newline += indent + f'fscanf(f, "%d", &input_{input_idx});\n'
                        newline += indent + 'fclose(f);\n\n'
                    if layer.get_attr('write_func'):
                        if func_layers_count == len([layer for layer in layers if layer.get_attr("write_func")]) - 1:
                            for output_idx in range(layer.get_attr("out_dim_val")):
                                newline += indent + f'default_t out{i}_{output_idx}[1];\n'
                        func_layers_count += 1

            elif '// hls-fpga-machine-learning input kernel' in line:
                newline = ''
                func_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        for input_idx in range(layer.get_attr("out_dim_val")):
                            newline += indent + indent + f'input_{input_idx}, \n'
                    elif layer.get_attr('write_func'):
                        if func_layers_count == len([layer for layer in layers if layer.get_attr("write_func")]) - 1:
                            for output_idx in range(layer.get_attr("out_dim_val")):
                                newline += indent + indent + f'out{i}_{output_idx}'
                                if output_idx < layer.get_attr("out_dim_val") - 1:
                                    newline += ',\n'
                                else:
                                    newline += '\n'
                        else:
                            func_layers_count += 1

            elif '// hls-fpga-machine-learning load weights' in line:
                newline = line
                for i, layer in enumerate(layers):
                    if layer.get_attr("write_weights"):
                        # Weights
                        newline += f'const default_t w{i}[{layer.get_attr("out_dim_key")}][{layer.get_attr("in_dim_key")}] = {{\n'
                        for idx_row, row in enumerate(layer.get_attr('fxp_weights')):
                            newline += indent
                            for idx_col, w in enumerate(row):
                                newline += f'{w}'
                                if idx_col < len(row) - 1:
                                    newline += ','
                            if idx_row < len(layer.get_attr("fxp_weights")) - 1:
                                    newline += ',\n'
                            else:
                                newline += '\n'
                        newline += '};\n'
                        # Bias
                        newline += f'const default_t b{i}[{layer.get_attr("out_dim_key")}] = {{\n'
                        newline += indent
                        for idx_b, b in enumerate(layer.get_attr("fxp_bias")):
                            newline += f'{b}'
                            if idx_b < len(layer.get_attr("fxp_bias")) - 1:
                                newline += ','
                        newline += '\n' + '};\n'

            # Just copy the line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_nnet_utils(self, model: ModelGraph) -> None:
        """Copy the nnet_utils, AP types headers to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/dynamatic/firmware/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        # TODO: future implementation of ap_types separation
        # # ap_types
        # filedir = os.path.dirname(os.path.abspath(__file__))

        # srcpath = os.path.join(filedir, '../templates/dynamatic/firmware/ap_types/')
        # dstpath = f'{model.config.get_output_dir()}/firmware/ap_types/'

        # if os.path.exists(dstpath):
        #     rmtree(dstpath)

        # copytree(srcpath, dstpath)

    def write_hls(self, model: ModelGraph) -> None:
        """Main writer function that calls multiple helper functions for writing the infrastrcture
        of the Dynamatic project. 

        Args:
            model (ModelGraph): hls4ml IR containining the neural network information.
        """
        self.write_project_dir(model)
        self.write_scripts(model)
        self.write_project_dynamatic(model)
        self.write_nnet_utils(model)