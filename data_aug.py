import SimpleITK as sitk
import numpy as np
import os
import glob
OUTPUT_DIR_img = "/home/brl/niftynet/data/CHOAS_liver/img_pre"
OUTPUT_DIR_label = "/home/brl/niftynet/data/CHOAS_liver/label_pre"
imgpath="/home/brl/niftynet/data/CHOAS_liver/img/*.nii.gz"
labelpath="/home/brl/niftynet/data/CHOAS_liver/label/*.nii.gz"
def parameter_space_regular_grid_sampling(*transformation_parameters):

    '''
    Create a list representing a regular sampling of the parameter space.     
    Args:
        *transformation_paramters : two or more numpy ndarrays representing parameter values. The order 
                                    of the arrays should match the ordering of the SimpleITK transformation 
                                    parameterization (e.g. Similarity2DTransform: scaling, rotation, tx, ty)
    Return:
        List of lists representing the regular grid sampling.
        
    Examples:
        #parameterization for 2D translation transform (tx,ty): [[1.0,1.0], [1.5,1.0], [2.0,1.0]]
        >>>> parameter_space_regular_grid_sampling(np.linspace(1.0,2.0,3), np.linspace(1.0,1.0,1))        
    '''
    return [[np.asscalar(p) for p in parameter_values] 
            for parameter_values in np.nditer(np.meshgrid(*transformation_parameters))]

def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parameterization uses the vector portion of a versor we don't have an 
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
        tx, ty, tz: numpy ndarrays with the translation values to use.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return [list(eul2quat(parameter_values[0],parameter_values[1], parameter_values[2])) + 
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]
def similarity3D_sampling(tx, ty, tz):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parameterization uses the vector portion of a versor we don't have an 
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
        tx, ty, tz: numpy ndarrays with the translation values to use.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return np.meshgrid(tx, ty, tz)
def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy 
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx     

    r[1,0] = sz*cy 
    r[1,1] = sz*sy*sx + cz*cx 
    r[1,2] = sz*sy*cx - cz*sx

    r[2,0] = -sy   
    r[2,1] = cy*sx             
    r[2,2] = cy*cx

    # Compute quaternion: 
    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs,0.0,atol): 
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom;
        qv[1] = (r[0,2] - r[2,0])/denom;
        qv[2] = (r[1,0] - r[0,1])/denom;
    return qv
#data generation
def resample_image(original_image, reference_image, T0,output_prefix, output_suffix,
                    interpolator = sitk.sitkNearestNeighbor, default_intensity_value = 0.0):
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system 
            to the original_image coordinate system.
    '''
    all_images = [] # Used only for display purposes in this notebook.
     
    # Augmentation is done in the reference image space, so we first map the points from the reference image space
    # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
    #T_all = sitk.Transform(T0)
    aug_image = sitk.Resample(original_image, reference_image, T0,
                              interpolator, default_intensity_value)
    sitk.WriteImage(aug_image, output_prefix + '_' + output_suffix)
     
    return aug_image# Used only for display purposes in this notebook.

def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters,
                    output_prefix, output_suffix,
                    interpolator = sitk.sitkLinear, default_intensity_value = 0.0):
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system 
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    '''
    all_images = [] # Used only for display purposes in this notebook.
   
    T_aug.SetParameters(transformation_parameters)        
    # Augmentation is done in the reference image space, so we first map the points from the reference image space
    # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
    T_all = sitk.Transform(T0)
    T_all.AddTransform(T_aug)
    aug_image = sitk.Resample(original_image, reference_image, T_all,
                              interpolator, default_intensity_value)
    sitk.WriteImage(aug_image, output_prefix + '_' + output_suffix)
     
    return aug_image# Used only for display purposes in this notebook.
def augment_images_flip(original_image, reference_image, T0,
                    output_prefix, output_suffix,
                    interpolator = sitk.sitkLinear, default_intensity_value = 0.0):
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system 
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    ''' 
        # Augmentation is done in the reference image space, so we first map the points from the reference image space
        # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
    T_all = sitk.Transform(T0)
    aug_image = sitk.Resample(original_image, reference_image, T_all,
                              interpolator, default_intensity_value)
    sitk.WriteImage(aug_image, output_prefix + '_' + '_.' + output_suffix)
     
     # Used only for display purposes in this notebook.
    return aug_image # Used only for display purposes in this notebook.

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#reading image
imageList=[]
labelList=[]
for name in glob.glob(imgpath):
    imageList.append(name)
for name in glob.glob(labelpath):
    labelList.append(name)
data_img=[]
data_label=[]
for im in range(len(imageList)):
 data_img.append(sitk.ReadImage(imageList[im]))
 data_label.append(sitk.ReadImage(labelList[im]))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Creating Reference domain
dimension = data_img[0].GetDimension()
# Physical image size corresponds to the largest physical size in the training se
#t, or any other arbitrary size.
reference_physical_size = np.zeros(dimension)
reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx else mx for sz,spc,mx in zip(data_img[0].GetSize(), data_img[0].GetSpacing(), reference_physical_size)]
# Create the reference image with a zero origin, identity direction cosine matrix
#and dimension
reference_origin = np.zeros(dimension)
reference_direction = np.identity(dimension).flatten()
# Select arbitrary number of pixels per dimension, smallest size that yields desired results (non-isotropic pixels)
#reference_size = [96]*dimension
#reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]
# Another possibility is that you want isotropic pixels, uncomment the following
#lines.
reference_size_x = 144
reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]
reference_image = sitk.Image(reference_size, data_img[0].GetPixelIDValue())
reference_image.SetOrigin(reference_origin)
reference_image.SetSpacing(reference_spacing)
reference_image.SetDirection(reference_direction)
# Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed po
#int's physical coordinates as
# this takes into account size, spacing and direction cosines. For the vast major
#ity of images the direction
# cosines are the identity matrix, but when this isn't the case simply multiplyin
#g the central index by the
# spacing will not yield the correct coordinates resulting in a long debugging se
#ssion.
reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))


#transformation
aug_transform = sitk.Similarity2DTransform() if dimension==2 else sitk.Similarity3DTransform()

all_images = []

# Transform which maps from the reference_image to the current img with the translation mapping the image
# origins to each other.

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
for index,img in enumerate(data_img):
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

#    flipped_transform = sitk.AffineTransform(dimension)    
#    flipped_transform.SetCenter(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

#    
#    flipped_transform.SetMatrix(matrix_flip[i])
#    centered_transform.AddTransform(flipped_transform)
    
    aug_transform.SetCenter(reference_center)
    if(index<=11):
        transformation_parameters_list=[np.pi/18.0,np.pi/18.0,np.pi/18.0,10,10,10,1.1]
    else:
        transformation_parameters_list=[-np.pi/18.0,-np.pi/18.0,-np.pi/18.0,-10,-10,-10,0.9]
       
#        transformation_parameters_list = similarity3D_parameter_space_regular_sampling(np.linspace(-np.pi/18.0,np.pi/18.0,2),
#                                                                                           np.linspace(-np.pi/18.0,np.pi/18.0,2),
#                                                                                           np.linspace(-np.pi/18.0,np.pi/18.0,2),
#                                                                                           np.linspace(-10,10,2),
#                                                                                           np.linspace(-10,10,2),
#                                                                                           np.linspace(-10,10,2),
#                                                                                           np.linspace(0.9,1.1,2))
        
##    transformation_parameters_list = similarity3D_sampling(np.linspace(-10,10,2),np.linspace(-10,10,2),np.linspace(-10,10,2))

    #generated_images = augment_images_spatial(img, reference_image, centered_transform, aug_transform, transformation_parameters_list, os.path.join(OUTPUT_DIR,'flip_aug'+str(i)), 'nii')

#        random_list = np.random.randint(0,127,size=10)
#        random_transformation_parameters_list = np.zeros(shape =(10,7))
#
#        for j in range(10):
#            random_transformation_parameters_list[j,:] = transformation_parameters_list[random_list[j]]

    generated_images = resample_image(img, reference_image, centered_transform, os.path.join(OUTPUT_DIR_img,str(130+index)+'_'+'img_liver'), '.nii.gz')
    generated_images = resample_image(data_label[index], reference_image, centered_transform, os.path.join(OUTPUT_DIR_label,str(130+index)+'_'+'label_liver'), '.nii.gz')
#    generated_images = augment_images_spatial(img, reference_image, centered_transform,aug_transform,transformation_parameters_list, os.path.join(OUTPUT_DIR_img,str(index)+'11'+'_'+'img_aug_brain'), '.nii.gz')
#    generated_images = augment_images_spatial(data_label[index], reference_image, centered_transform,aug_transform,transformation_parameters_list, os.path.join(OUTPUT_DIR_label,str(index)+'11'+'_'+'label_aug_seg'), '.nii.gz')


#---------------------------------------------------------------------------------------
#-------------------------------- Flipping ---------------------------------------------
#--------------------------------------------------------------------------------------

#flipped_images = []
#OUTPUT_DIR_imgf = "E:/J/Data/Synapse/Data_Aug/VNetBTCV/flipped/img"
#OUTPUT_DIR_labelf = "E:/J/Data/Synapse/Data_Aug/VNetBTCV/flipped/label"
#imgpath="E:/J/Data/Synapse/Data_Aug/VNetBTCV/augmented/img/*.nii.gz"
#labelpath="E:/J/Data/Synapse/Data_Aug/VNetBTCV/augmented/label/*.nii.gz"
## data reading
#imageList=[]
#labelList=[]
#for name in glob.glob(imgpath):
#    imageList.append(name)
#for name in glob.glob(labelpath):
#    labelList.append(name)
#data_img=[]
#data_label=[]
#for im in range(len(imageList)):
# data_img.append(sitk.ReadImage(imageList[im]))
# data_label.append(sitk.ReadImage(labelList[im]))
#
## Compute the transformation which maps between the reference and current image (same as done above).
#for index,img in enumerate(data_img):
#    transform = sitk.AffineTransform(dimension)
#    transform.SetMatrix(img.GetDirection())
#    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
#    centering_transform = sitk.TranslationTransform(dimension)
#    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
#    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
#    centered_transform = sitk.Transform(transform)
#    centered_transform.AddTransform(centering_transform)
#        
#    flipped_transform = sitk.AffineTransform(dimension)    
#    flipped_transform.SetCenter(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
#    matrix_flip = [[1,0,0,0,1,0,0,0,-1],[1,0,0,0,-1,0,0,0,1],[-1,0,0,0,1,0,0,0,1]]
#    
#    if(index<=15):
#        flipped_transform.SetMatrix(matrix_flip[0])
#    if((index>=16) & (index<=38)):
#        flipped_transform.SetMatrix(matrix_flip[1])
#    else:
#        flipped_transform.SetMatrix(matrix_flip[2])
#        
#        
#    
#    centered_transform.AddTransform(flipped_transform)
#    generated_images = augment_images_flip(img, reference_image, centered_transform, os.path.join(OUTPUT_DIR_imgf,'img_aug'+str(index)), 'nii.gz')
#    generated_images = augment_images_flip(data_label[index], reference_image,centered_transform, os.path.join(OUTPUT_DIR_labelf,'label_aug'+str(index)), 'nii.gz')    
    

