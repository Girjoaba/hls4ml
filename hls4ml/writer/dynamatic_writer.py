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
            model (ModelGraph): the hls4ml model.
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


    def write_project_dynamatic(self, model: ModelGraph) -> None:
        """Write the main architecture source file (myproject.x)

        Args:
            model (ModelGraph): the hls4ml model.
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
                    #     if layer.class_name == 'Input':
                    #         newline += f'#define INPUT_SIZE {layer.get_attr("out_dim_val")}\n'
                    #     else:
                    #         newline += f'#define {layer.get_attr("in_dim_key")} {layer.get_attr("in_dim_val")}\n'
                        newline += f'#define {layer.get_attr("out_dim_key")} {layer.get_attr("out_dim_val")}\n'

            elif '// hls-fpga-machine-learning architecture arguments' in line:
                newline = ''
                func_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        newline += indent + f'default_t input[{layer.get_attr("out_dim_key")}], \n'
                    elif layer.get_attr('write_func'):
                        newline += indent + f'default_t out{i}[{layer.get_attr("out_dim_key")}]'
                        if func_layers_count < len([layer for layer in layers if layer.get_attr("write_func")]) - 1:
                            newline += ',\n'
                            func_layers_count += 1
                        else:
                            newline += '\n'

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line
                prev_var = 'input'
                for i, layer in enumerate(layers):
                    if layer.get_attr('write_func'):
                        if layer.get_attr('write_weights'):
                            newline += indent + f'dense_accum_t acc{i};\n'
                            newline += indent + f'default_t tmp{i};\n'
                            newline += indent + f'{layer.get_attr("func_call")}({prev_var}, out{i}, {layer.get_attr("in_dim_key")}, {layer.get_attr("out_dim_key")}, w{i}, b{i}, acc{i}, tmp{i});\n'
                            prev_var = f'out{i}'
                        # else:
                        #     newline += indent + f'let z{i} = {layer.get_attr("func_call")}({prev_var});\n'
                        #     prev_var = f'out{i}'


            elif '// hls-fpga-machine-learning input init' in line:
                newline = line
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        newline += indent + f'default_t input[{layer.get_attr("out_dim_key")}];\n'
                        newline += indent + f'FILE *f = fopen("input.txt", "r");\n'
                        newline += indent + f'for (int i=0; i < {layer.get_attr("out_dim_key")}; ++i) {{\n'
                        newline += indent + indent + 'fscanf(f, "%d", &input[i]);\n'
                        newline += indent + '}\n\n'
                        newline += indent + 'fclose(f);\n\n'
                    elif layer.get_attr('write_func'):
                        newline += indent + f'default_t out{i}[{layer.get_attr("out_dim_key")}];\n'

            elif '// hls-fpga-machine-learning input kernel' in line:
                newline = ''
                func_layers_count = 0
                for i, layer in enumerate(layers):
                    if layer.class_name == 'Input':
                        newline += indent + indent + f'input, \n'
                    elif layer.get_attr('write_func'):
                        newline += indent + indent + f'out{i}'
                        if func_layers_count < len([layer for layer in layers if layer.get_attr("write_func")]) - 1:
                            newline += ',\n'
                            func_layers_count += 1
                        else:
                            newline += '\n'

            elif '// hls-fpga-machine-learning load weights' in line:
                newline = line
                for i, layer in enumerate(layers):
                    if layer.get_attr("write_weights"):
                        # Weights
                        newline += f'const default_t w{i}[{layer.get_attr("in_dim_key")}][{layer.get_attr("out_dim_key")}] = {{\n'
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

            # Just copy line
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

        # ap_types
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/dynamatic/firmware/ap_types/')
        dstpath = f'{model.config.get_output_dir()}/firmware/ap_types/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

    def write_hls(self, model: ModelGraph) -> None:

        self.write_project_dir(model)
        self.write_scripts(model)
        self.write_project_dynamatic(model)
        self.write_nnet_utils(model)