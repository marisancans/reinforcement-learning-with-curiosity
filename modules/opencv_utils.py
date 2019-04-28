from modules.torch_utils import to_numpy            
            
             
            
            
def debug_encoded_states(pred_next_state_t, next_state_t):
    pred = np.array(to_numpy(pred_next_state) * 255, dtype = np.uint8)
    target = np.array(to_numpy(next_state_t * 255, dtype = np.uint8))
    
    pred = np.stack((pred,)*3, axis=-1)
    target = np.stack((target,)*3, axis=-1)

    delim = np.full((pred.shape[0], 1, 3), (0, 0, 255), dtype=np.uint8)
    pred = np.concatenate((pred, delim), axis=1)
    img = np.concatenate((pred, target), axis=1)
    
    cv2.namedWindow('features', cv2.WINDOW_NORMAL)
    cv2.imshow('features', img)
    cv2.waitKey(1)

def debug_sequence():
    features = np.array(self.states_sequence * 255, dtype = np.uint8)
    
    #img = np.stack((img,)*3, axis=-1)
    features = np.concatenate(features, axis=1)
    cv2.namedWindow('sequence', cv2.WINDOW_NORMAL)
    cv2.imshow('sequence', features)
    
    cv2.waitKey(1)
    # 
    #self.env.render()

    key = self.args.debug_activations[0]
    features = self.encoder_model.activations[key]
    features = features.squeeze(0).cpu().detach().numpy()
    features = np.array(features * 255, dtype = np.uint8)

    col_count = 10
    height = features.shape[1]
    width = features.shape[2]
    blank_count = col_count - (features.shape[0] % col_count)
    
    # Fill missing feature maps with zeros
    for i in range(blank_count):
        blank = np.zeros(shape=(1, features.shape[1], features.shape[2]), dtype=np.uint8)
        features = np.concatenate((features, blank))

    # Merge all feature maps into 2D image
    features = np.reshape(features, newshape=(-1, col_count, features.shape[1], features.shape[2]))
    row_count = features.shape[0]
    features = np.concatenate(features, axis=1)
    features = np.concatenate(features, axis=1)

    img = np.stack((features,)*3, axis=-1) # Make RGB
    
    # Make grid
    for c, a, s in [[col_count, 1, width], [row_count, 0, height]]:
        for i in range(1, c):
            pos = (s * i + i) - 1
            img = np.insert(img, pos, values=(229, 0, 225), axis=a) 

    cv2.namedWindow('activations', cv2.WINDOW_NORMAL)
    cv2.imshow('activations', img)
    cv2.waitKey(1)