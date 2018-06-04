function argMax(array) {
    return array.reduce((acc, cval, cind, arr) => cval > arr[acc] ? cind : acc, 0);
}

/**
 * Sum of array elements over a given axis.
 *
 * @param {Matrix} matrix Input array
 * @param {number} axis Axis along which a sum is performed.
 * @return {Array} A vector arry with the specified axis removed containing the sum.
 */
function sumAxis(matrix, axis) {
    if (axis===0) { 
        var sum = math.zeros(matrix.dimensions().cols);
        
        for (var i=0; i < matrix.dimensions().rows; i++){
            for (var j=0; j < matrix.dimensions().cols; j++){
                sum._data[j] += matrix.elements[i][j];
                //console.log('i:' + i + '  j:' + j + '  ' + matrix.elements[i][j]);
            }
        }
    } else {
        var sum = math.zeros(matrix.dimensions().rows);
        
        for (var i=0; i < matrix.dimensions().rows; i++){
            for (var j=0; j < matrix.dimensions().cols; j++){
                sum._data[i] += matrix.elements[i][j];
            }
        }
    }

    return sum;
}

function sumVector(vector) {
    return vector.reduce(function(acc, val) { return acc + val; });
}

function maxOf2DArray(array) {
    var maxRow = array.map(function(row){ return Math.max.apply(Math, row); });
    return Math.max.apply(null, maxRow);
}

function minOf2DArray(array) {
    var minRow = array.map(function(row){ return Math.min.apply(Math, row); });
    return Math.min.apply(null, minRow);
}

var rmin = -1;
var rmax = 1;

function computeHeatmap (forwardPassOutputs) {

    // set relevance of output layer to zeros except for neuron corresponding to prediction
    var pred = argMax(forwardPassOutputs.output.elements);
    var R_final = Vector.Zero(10);
    R_final.elements[pred] = 1;
    
    var R_hidden2 = lrp_dense(forwardPassOutputs.hidden2, final_weights, final_biases, R_final);
    var R_hidden1 = lrp_dense(forwardPassOutputs.hidden1, hidden_weights_2, hidden_biases_2, R_hidden2);

    // compute relevances for downsampling 2 as flat array (shape: 400)
    var R_downsampling_2 = lrp_dense(forwardPassOutputs.downsamplingLayer2_flat, hidden_weights_1, 
        hidden_biases_1, R_hidden1);

        
    // relevances for convolutional layer 2
    var R_convolution_2 = [];
    var size = 10;
    for (var f=0; f<nConvFilters_2; f++) {
        R_convolution_2[f] = Matrix.Zero(size, size);

        for (var i=0; i<size; i++) {
            for (var j=0; j<size; j++) {
                var pooled_index = f*(size/2)*(size/2) + math.floor(i/2) * filterSize_2 + math.floor(j/2);
                //R_convolution_2[f].elements[i][j] = pooled_index; // for checking indices
                
                if (forwardPassOutputs.convLayer2[f][i+2][j+2] == forwardPassOutputs.downsamplingLayer2_flat.elements[pooled_index]) {
                    // no normalization necessary as pooling filters are not overlapping
                    R_convolution_2[f].elements[i][j] = R_downsampling_2.elements[pooled_index];
                }
            }
        }
    }

    // relevance propagation from convlayer 2 to  downsampling 1 (i.e. convolution 2 backwards)
    var Zsum = forwardPassOutputs.convLayer2_activations;
    var inputImageSize = 14;
    var R_downsampling_1 = [];
    for (k=0; k<nConvFilters_1; k++) {
        R_downsampling_1[k] = Matrix.Zero(inputImageSize, inputImageSize);
    }
    var halfFilter = math.floor(filterSize_2/2);
    for (f=0; f<nConvFilters_2; f++) {
		for (i=halfFilter; i<inputImageSize-halfFilter; i++) {
			for (j=halfFilter; j<inputImageSize-halfFilter; j++) {
				var keeperCount = 0;
				for (k=0; k<nConvFilters_1; k++) {
					if (keepers.e(k+1,f+1)){ // wenn verbunden mit DownSamplingLayer1, dann..
						image = forwardPassOutputs.downsamplingLayer1[k];
						for (m=-halfFilter; m<=halfFilter; m++) {
							for (n=-halfFilter; n<=halfFilter; n++) {								
                                var activation = image[i+m][j+n] * conv_nodes[1][f][keeperCount].e(m+halfFilter+1,n+halfFilter+1);
                                var denominator = Zsum[f][i-halfFilter][j-halfFilter] + conv_biases_2.e(f+1);
                                if (denominator > 0) {
                                    denominator += 1e-12;
                                } else {
                                    denominator -= 1e-12; 
                                }
                                var r = activation / denominator * R_convolution_2[f].elements[i-halfFilter][j-halfFilter];
                                R_downsampling_1[k].elements[i-halfFilter+m+halfFilter][j-halfFilter+n+halfFilter] += r;       
							}
						}
						keeperCount++;
					}
				}
			}
		}
	}

    
    // relevance propagation from downsampling 1 to convolution 1
    var R_convolution_1 = [];
    var size = 28; // size of convolution 1 output images
    for (var f=0; f<nConvFilters_1; f++) {
        R_convolution_1[f] = Matrix.Zero(size, size);

        for (var i=0; i<size; i++) {
            for (var j=0; j<size; j++) {

                if (forwardPassOutputs.convLayer1[f][i+2][j+2] == forwardPassOutputs.downsamplingLayer1[f][math.floor(i/2)][math.floor(j/2)]) {
                    // no normalization necessary as pooling filters are not overlapping
                    R_convolution_1[f].elements[i][j] = R_downsampling_1[f].elements[math.floor(i/2)][math.floor(j/2)];
                }
            }
        }
    }

    // relevance propagation from convolution 1 to input image
    var inputImageSize = 32;
    var R_input = Matrix.Zero(inputImageSize,inputImageSize);
    var halfFilter = math.floor(filterSize_1/2);
    for (f=0; f<nConvFilters_1; f++) {
        for (i=halfFilter; i<inputImageSize-halfFilter; i++) {
			for (j=halfFilter; j<inputImageSize-halfFilter; j++) {
                
                for (m=-halfFilter; m<=halfFilter; m++) {
					for (n=-halfFilter; n<=halfFilter; n++) {
                        var activation = (forwardPassOutputs.inputLayer[i+m][j+n] )*  conv_nodes[0][f].e(m+halfFilter+1,n+halfFilter+1);
                        var denominator = forwardPassOutputs.convLayer1_activations[f][i-halfFilter][j-halfFilter] + conv_biases_1.e(f+1);
                        if (denominator > 0) {
                            denominator += 1e-12;
                        } else {
                            denominator -= 1e-12; 
                        }
                        var r = activation / denominator * R_convolution_1[f].elements[i-halfFilter][j-halfFilter];
                        R_input.elements[i-halfFilter+m+halfFilter][j-halfFilter+n+halfFilter] += r;       
					}
				}				
			}
		}
	}

    rmax = maxOf2DArray(R_input.elements);
    rmin = minOf2DArray(R_input.elements);
    var count = layerStartIndices.heatmap;
    for (var i=0; i<inputImageSize; i++) {
        for (var j=0; j<inputImageSize; j++) {
            var r = R_input.elements[i][j];
            // r = (r - rmin) / (rmax - rmin);
            allNodeOutputs[count] = r;
            
            allNodeNums[count] = count - layerStartIndices.heatmap + 1;
            count++;
        }
    }


    // make variables available in web console
    window.forwardPassOutputs = forwardPassOutputs; 
    window.R_final = R_final;
    window.R_hidden2 = R_hidden2;
    window.R_hidden1 = R_hidden1;
    window.R_downsampling_2 = R_downsampling_2;
    window.R_convolution_2 = R_convolution_2;
    window.R_downsampling_1 = R_downsampling_1;
    window.R_convolution_1 = R_convolution_1;
    window.R_input = R_input;
}


function lrp_dense(inputs, weights, biases, R_j) {
    var len_i = inputs.dimensions();
    var len_j = R_j.dimensions();
    var R_i = Vector.Zero(len_i);
    var Z_ij = Matrix.Zero(len_i, len_j);

    // // compute activations
    for (var i=0; i < len_i; i++) {
        for (var j=0; j < len_j; j++) {
            Z_ij.elements[i][j] = inputs.e(i+1) * weights.e(j+1, i+1);
        }
    }
    var Zs = sumAxis(Z_ij, axis=0);

    for (var i=0; i < len_i; i++) {
        for (var j=0; j < len_j; j++) {
            var z = Z_ij.elements[i][j] / (Zs._data[j] + biases.elements[j]);
            z *= R_j.e(j+1);            
            
            R_i.elements[i] += z;
        }
    }

    return R_i;
}