import csv
import joblib
import json
import numpy as np
import os


class SKLearnManager:
    """
    Comprehensive model manager that handles saving, loading, and inference for
    sklearn models with support for multiple backends (sklearn/joblib, ONNX,
    and version-independent custom implementation).
    
    Attributes:
        filename (str): Base filename for the model
        model_type (str): Type of model backend ('sklearn', 'onnx', or 'custom')
        model: The loaded model object
    """
    
    VALID_TYPES = ['sklearn', 'onnx', 'custom']
    
    def __init__(self, filename, model_type='sklearn'):
        """
        Initialize the model manager.
        
        Args:
            filename (str): Base filename for the model (without extension)
            model_type (str): Backend to use ('sklearn', 'onnx', or 'custom')
        """
        self.filename = filename
        if model_type.lower() not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid model_type. Must be one of: {', '.join(self.VALID_TYPES)}"
                )
        
        self.model_type = model_type.lower()
        self.model = None
    
    def save(self, model, n_features=None) :
        """
        Save a model using the specified backend.
        
        Args:
            model: Trained sklearn model to save
            X_sample: Sample input data (required for ONNX conversion)
        """
        if self.model_type == 'sklearn':
            self._save_sklearn(model)
        elif self.model_type == 'onnx':
            self._save_onnx(model, n_features)
        elif self.model_type == 'custom':
            self._save_custom(model)
    
    def load(self):
        """
        Load the model using the specified backend.
        """
        if self.model_type == 'sklearn':
            self.model = self._load_sklearn()
        elif self.model_type == 'onnx':
            self.model = self._load_onnx()
        elif self.model_type == 'custom':
            self.model = self._load_custom()
    
    def predict(self, X):
        """
        Make predictions using the loaded model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if self.model_type == 'sklearn':
            return self.model.predict(X)
        elif self.model_type == 'onnx':
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            return self.model.run(
                [output_name], 
                {input_name: X.astype(np.float32)}
            )[0]
        elif self.model_type == 'custom':
            return self.model.predict(X)
    
    def predict_individual_trees(self, X, n_outputs):
        """
        Get predictions from individual trees (for ensemble models).
        Only works with 'sklearn' or 'custom' model types.
        
        Args:
            X: Input features
            
        Returns:
            Individual tree predictions or None if not supported
        """
        if self.model_type == 'sklearn':
            #if n_outputs is None:
            #    if len(self.model.estimators_[0].predict(X).shape) == 1:
            #        n_outputs = 1
            #    else:
            #        n_outputs = self.model.estimators_[0].predict(X).shape[1]
            if len(X.shape) == 1:
                X = X[np.newaxis]
            elif len(X.shape) > 2:
                assert False
            n_samples = X.shape[0]
            n_estimators = len(self.model.estimators_)

            # Initialize predictions array with appropriate dimensions
            preds = np.zeros((n_samples, n_outputs, n_estimators))
            for tree_index, tree in enumerate(self.model.estimators_):
                if n_outputs == 1:
                    preds[:, :, tree_index] = tree.predict(X)[:, np.newaxis]
                else:
                    preds[:, :, tree_index] = tree.predict(X)
            return preds
        elif self.model_type == 'custom':
            return self.model.predict_individual_trees(X, n_outputs)
        else:
            raise ValueError("Individual tree predictions not supported for ONNX models")
    
    def _save_sklearn(self, model):
        """Save model using joblib"""
        joblib.dump(model, f"{self.filename}.joblib")
    
    def _save_onnx(self, model, n_features):
        """Save model in ONNX format"""
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Define input type
        initial_types = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX with appropriate options
        #is_classifier = hasattr(model, 'classes_')
        #options = {id(model): {'zipmap': False}} if is_classifier else None
        options = {type(model): {'output_type': 'tensor(float)'}}
        onnx_model = convert_sklearn(
            model, 
            initial_types=initial_types,
            target_opset=15,
            #options=options
        )
        
        # Save ONNX model
        with open(f"{self.filename}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
    
    def _save_custom(self, model) -> None:
        """Save model in custom version-independent format, supporting multi-output trees"""
        # Save with joblib as backup
        joblib.dump(model, f"{self.filename}.joblib")
        
        # Extract and save essential tree information
        if not hasattr(model, 'estimators_'):
            raise ValueError("Custom format only supports ensemble models with estimators_")
            
        forest_data = {
            'n_estimators': len(model.estimators_),
            'n_features': model.n_features_in_,
            'trees': []
        }
        
        # Extract each tree's node structure
        for tree in model.estimators_:
            tree_dict = {
                'nodes': [],
                'values': tree.tree_.value.tolist()
            }
            
            # Convert tree structure to list of node dictionaries
            for node_id in range(tree.tree_.node_count):
                node = {
                    'id': node_id,
                    'left_child': int(tree.tree_.children_left[node_id]),
                    'right_child': int(tree.tree_.children_right[node_id]),
                    'feature': int(tree.tree_.feature[node_id]),
                    'threshold': float(tree.tree_.threshold[node_id]),
                    'value_index': node_id
                }
                tree_dict['nodes'].append(node)
            
            forest_data['trees'].append(tree_dict)
        
        # Save as JSON
        with open(f"{self.filename}_trees.json", 'w') as f:
            json.dump(forest_data, f)
    
    def _load_sklearn(self):
        """Load sklearn model using joblib"""
        model_path = f"{self.filename}.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)
    
    def _load_onnx(self):
        """Load ONNX model"""
        import onnxruntime
        model_path = f"{self.filename}.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        return onnxruntime.InferenceSession(model_path, sess_options)
    
    def _load_custom(self):
        """Load custom version-independent forest predictor"""
        model_path = f"{self.filename}_trees.json"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'r') as f:
            forest_data = json.load(f)
        
        return self.VersionIndependentForestPredictor(forest_data)
    
    class VersionIndependentForestPredictor:
        """Custom tree-based model implementation that works across sklearn versions"""
        
        def __init__(self, forest_data):
            self.n_estimators = forest_data['n_estimators']
            self.n_features = forest_data['n_features']
            self.trees = forest_data['trees']
        
        def predict(self, X, n_outputs=None):
            """
            Get ensemble prediction, supporting both single-output and multi-output trees.
            
            Args:
                X: Input features
                
            Returns:
                Ensemble predictions
            """
            predictions = self.predict_individual_trees(X, n_outputs)
            
            # For multi-output, we need to average over the first axis (trees)
            if predictions.ndim == 3:  # (n_estimators, n_samples, n_outputs)
                return np.mean(predictions, axis=0)  # -> (n_samples, n_outputs)
            else:  # (n_estimators, n_samples)
                return np.mean(predictions, axis=0)  # -> (n_samples,)
        
        def predict_individual_trees(self, X, n_outputs=None):
            """
            Get predictions from each individual tree, supporting both single-output
            and multi-output trees.
            
            Args:
                X: Input features of shape (n_samples, n_features)
                
            Returns:
                Individual tree predictions of shape:
                - For single output: (n_estimators, n_samples)
                - For multi output: (n_estimators, n_samples, n_outputs)
            """
            if X.ndim == 1:
                X = X.reshape(1, -1)
                
            n_samples = X.shape[0]
            
            # Determine output dimensionality from the first tree's values
            if n_outputs is None:
                first_leaf_value = self.trees[0]['values'][0]
                n_outputs = len(first_leaf_value)
            # Initialize predictions array with appropriate dimensions
            if n_outputs == 1:
                individual_preds = np.zeros((n_samples, self.n_estimators))
            else:
                individual_preds = np.zeros((n_samples, n_outputs, self.n_estimators))

            for tree_idx, tree in enumerate(self.trees):
                nodes = tree['nodes']
                values = tree['values']
                
                for sample_idx, sample in enumerate(X):
                    # Start at root and traverse tree
                    node_id = 0
                    
                    # Traverse until we reach a leaf
                    while True:
                        node = nodes[node_id]
                        
                        # Check if we're at a leaf node
                        if node['left_child'] == -1:  # Leaf node
                            leaf_value = values[node['value_index']]
                            
                            if n_outputs == 1:
                                individual_preds[sample_idx, tree_idx] = leaf_value[0]
                            else:
                                for output_index in range(n_outputs):
                                    individual_preds[sample_idx, output_index, tree_idx] = leaf_value[output_index][0]
                            break
                            
                        # If not leaf, go left or right based on feature comparison
                        if sample[node['feature']] <= node['threshold']:
                            node_id = node['left_child']
                        else:
                            node_id = node['right_child']

            return individual_preds


class NeuralNetworkManager:
    def __init__(self, model_name="model", save_dir="saved_models"):
        """
        Initialize the model manager.
        
        Args:
            model: A Keras model instance (optional)
            model_name: Name to use when saving the model
            save_dir: Directory to save models
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.onnx_session = None
        self.quantized_onnx_session = None
        if 'KERAS_BACKEND' in os.environ.keys():
            if os.environ["KERAS_BACKEND"] == 'torch':
                self.convert_to_onnx = self._convert_to_onnx_pytorch
            elif os.environ["KERAS_BACKEND"] == 'tensorflow':
                self.convert_to_onnx = self._convert_to_onnx_tensorflow
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
    def _get_keras_weights_path(self):
        """Get the path for saving Keras weights."""
        return os.path.join(self.save_dir, f"{self.model_name}.weights.h5")
    
    def _get_onnx_path(self):
        """Get the path for saving the ONNX model."""
        return os.path.join(self.save_dir, f"{self.model_name}.onnx")
    
    def _get_quantized_onnx_path(self):
        """Get the path for saving the quantized ONNX model."""
        return os.path.join(self.save_dir, f"{self.model_name}_quantized.onnx")
    
    def save_keras_weights(self, model):
        """Save the Keras model weights."""
        weights_path = self._get_keras_weights_path()
        model.save_weights(weights_path)
        print(f"Keras model weights saved to {weights_path}")
        return weights_path
    
    def _convert_to_onnx_pytorch(self, model, example_inputs=None, input_signature=None):
        """
        Convert the Keras model to ONNX format.
        
        Args:
            input_signature: Optional input signature for the model
                            (e.g., [keras.InputSpec(dtype='float32', shape=(None, 224, 224, 3)])
        """
        import torch
        onnx_path = self._get_onnx_path()
        # Convert the model to ONNX
        if type(example_inputs) == tuple:
            dummy_input = [
                np.random.random((1, inputs.shape[1])).astype(np.float32) for inputs in example_inputs
                ]
            dummy_input_tensor = [
                torch.from_numpy(inputs) for inputs in dummy_input
                ]
            input_names = [f'input_{i}' for i in range(len(example_inputs))]
        else:
            dummy_input = np.random.random((1, example_inputs.shape[1])).astype(np.float32)
            dummy_input_tensor = torch.from_numpy(dummy_input)
            input_names=['input']
        # No dynamic axes - everything is static
        # This implies that the batch size is always one
        torch.onnx.export(
            model=model,
            args=dummy_input_tensor,
            f=onnx_path,
            opset_version=13,
            verbose=False,
            export_params=True,
            dynamic_axes=None, 
            do_constant_folding=True,
            output_names=['output'],
            input_names=input_names
            )
        print(f"ONNX model saved to {onnx_path}")
        return onnx_path

    def _convert_to_onnx_tensorflow(self, model, example_inputs=None, input_signature=None):
        """
        Convert the Keras model to ONNX format.
        
        Args:
            input_signature: Optional input signature for the model
                            (e.g., [keras.InputSpec(dtype='float32', shape=(None, 224, 224, 3)])
        """
        import tf2onnx
        import keras
        import onnx
        
        onnx_path = self._get_onnx_path()
        
        # If input signature is not provided, try to infer it
        if input_signature is None:
            # Try to infer from model inputs
            if hasattr(model, 'inputs') and model.inputs:
                shapes = []
                for inp in model.inputs:
                    shape = inp.shape
                    # Handle different backends by getting shape as list
                    if hasattr(shape, 'as_list'):
                        shape = shape.as_list()
                    # Get dtype in a backend-agnostic way
                    if hasattr(inp, 'dtype'):
                        if hasattr(inp.dtype, 'as_numpy_dtype'):
                            dtype = inp.dtype.as_numpy_dtype
                        else:
                            dtype = inp.dtype
                    else:
                        dtype = 'float32'  # Default to float32
                    shapes.append(keras.InputSpec(dtype=dtype, shape=shape))
                input_signature = shapes
            else:
                raise ValueError("Please provide input_signature for ONNX conversion")
        
        # Convert the model to ONNX
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        # Save the ONNX model
        onnx.save(model_proto, onnx_path)
        print(f"ONNX model saved to {onnx_path}")
        return onnx_path
    
    def quantize_onnx(self, method="dynamic", calibration_data=None):
        import onnx
        import onnxruntime.quantization
        """
        Perform 8-bit quantization on the ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model (optional, will use default if not provided)
            method: Quantization method - "dynamic" or "static"
            calibration_data: Required for static quantization - representative dataset
                             as a list of numpy arrays matching model inputs
        """
        onnx_path = self._get_onnx_path()
            
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        quantized_path = self._get_quantized_onnx_path()
        
        if method.lower() == "dynamic":
            model = onnx.load(onnx_path)
            import onnxruntime.quantization.preprocess
            onnxruntime.quantization.preprocess.quant_pre_process(
                input_model=onnx_path,
                output_model_path=onnx_path.replace('.onnx', '_preprocessed.onnx'),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
                )
            print(f"Preprocessed ONNX model saved to {onnx_path.replace('.onnx', '_preprocessed.onnx')}")
            # Perform dynamic quantization to 8-bit
            onnxruntime.quantization.quantize_dynamic(
                model_input=onnx_path.replace('.onnx', '_preprocessed.onnx'),
                model_output=quantized_path,
                per_channel=True,
                reduce_range=False,
                weight_type=onnxruntime.quantization.QuantType.QUInt8,
            )
            print(f"Dynamic quantized ONNX model saved to {quantized_path}")
        
        elif method.lower() == "static":
            # This is AI generated code that does not work.
            assert False
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration_data")
                
            # For static quantization, you would need to implement a DataReader
            # This is a simplified example - you may need to adapt it for your specific model
            from onnxruntime.quantization.calibrate import CalibrationDataReader
            
            class MyCalibrationDataReader(CalibrationDataReader):
                def __init__(self, input_data, input_name="input"):
                    self.input_data = input_data
                    self.datasize = len(input_data)
                    self.current_index = 0
                    self.input_name = input_name
                    
                def get_next(self):
                    if self.current_index >= self.datasize:
                        return None
                    
                    # Explicitly convert to float32
                    data = self.input_data[self.current_index].astype(np.float32)
                    result = {self.input_name: data}
                    self.current_index += 1
                    return result
            
            # Load the model to get input names
            model = onnx.load(onnx_path)
            # Create the calibration data reader
            data_reader = MyCalibrationDataReader(
                calibration_data.astype(np.float32),
                model.graph.input[0].name
                )
            
            # Perform static quantization
            onnxruntime.quantization.quantize_static(
                model_input=onnx_path,
                model_output=quantized_path,
                calibration_data_reader=data_reader,
                quant_format=onnxruntime.quantization.QuantFormat.QDQ,
                per_channel=True,
                reduce_range=False,
                activation_type=onnxruntime.quantization.QuantType.QUInt8,
                weight_type=onnxruntime.quantization.QuantType.QUInt8,
            )
            print(f"Static quantized ONNX model saved to {quantized_path}")
        
        else:
            raise ValueError(f"Unknown quantization method: {method}. Use 'dynamic' or 'static'.")
            
        return quantized_path
    
    def save_model(self, model, input_signature=None, quant_method="dynamic", calibration_data=None):
        """
        Save the model in both Keras weights and quantized ONNX formats.
        
        Args:
            input_signature: Optional input signature for ONNX conversion
            quant_method: Quantization method - "dynamic" or "static"
            calibration_data: Required for static quantization
        """
        # Save Keras weights
        self.save_keras_weights(model)
        
        # Convert to ONNX
        onnx_path = self.convert_to_onnx(model, input_signature)
        
        # Quantize ONNX model
        quantized_path = self.quantize_onnx(onnx_path, method=quant_method, calibration_data=calibration_data)
        
        return {
            "keras_weights": self._get_keras_weights_path(),
            "onnx": onnx_path,
            "quantized_onnx": quantized_path
        }
    
    def load_keras_model(self, model):
        """
        Load the Keras model from saved weights.
        
        Args:
            model: A Keras model with the same architecture as the saved weights.        
        Returns:
            The loaded Keras model
        """
        self.model = model
        weights_path = self._get_keras_weights_path()
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Keras weights not found at {weights_path}")
        self.model.load_weights(weights_path)
        #print(f"Loaded Keras model weights from {weights_path}")
        return self.model
    
    def load_onnx_model(self, quantized=True):
        """
        Load the ONNX model as an ONNX Runtime InferenceSession.
        
        Args:
            quantized: Whether to load the quantized model (True) or the regular ONNX model (False)
        
        Returns:
            An ONNX Runtime InferenceSession
        """
        import onnxruntime
        if quantized:
            onnx_path = self._get_quantized_onnx_path()
        else:
            onnx_path = self._get_onnx_path()
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        # Create an ONNX Runtime inference session
        # The code is designed to run with MPI. Threading creates significant competition
        # between processes for resources. The ensemble testing parallelization runtime
        # was signficantly degraded by the threading. There should be a smarter way to
        # manage this than draconian thread bans, but this fixes the issue.
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, sess_options)
        #print(f"Loaded ONNX model from {onnx_path}")
        return self.onnx_session
    
    def predict_with_onnx(self, input_data):
        # Run inference
        outputs = self.onnx_session.run(None, input_data)
        return outputs


def save_standard_scaler(scaler, filename):
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'n_features_in': int(scaler.n_features_in_),
        'n_samples_seen': int(scaler.n_samples_seen_)
        }
    with open(filename, 'w') as f:
        json.dump(scaler_params, f)


def load_standard_scaler(filename):
    from sklearn.preprocessing import StandardScaler
    with open(filename, 'r') as f:
        params = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(params['mean'])
    scaler.scale_ = np.array(params['scale'])
    scaler.n_features_in_ = params['n_features_in']
    scaler.n_samples_seen_ = params['n_samples_seen']
    return scaler


def write_params(params, filename):
    with open(filename, 'w') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=params.keys())
        writer.writeheader()
        writer.writerow(params)


def read_params(filename):
    with open(filename, 'r') as params_file:
        reader = csv.DictReader(params_file)
        for row in reader:
            params = row
    return params

