# Set-Executionpolicy RemoteSigned

$bw2color_image='D:\coding\python\bw-colorization\bw2color_image.py'
$imagepath='D:\coding\python\bw-colorization\images\'
$modelpath='D:\coding\python\bw-colorization\model\'

# write-host $bw2colorpath
# write-host $imagepath
# write-host $modelpath

$imagename = read-host 'Enter image file name (with ext)'

#$runparameter =$bw2colorpath' --image' $imagepath $imagename '--prototxt' $modelpath'colorization_deploy_v2.prototxt' '--model' $modelpath 'colorization_release_v2.caffemodel'' --points '$modelpath'pts_in_hull.npy'


# write-host $bw2colorpath
# write-host $imagepath$imagename 
C:\ProgramData\Anaconda3\python.exe $bw2color_image '--image' $imagepath$imagename '--prototxt' $modelpath'colorization_deploy_v2.prototxt' '--model' $modelpath'colorization_release_v2.caffemodel' '--points' $modelpath'pts_in_hull.npy'



#python $runparameter

# python D:\coding\python\bw-colorization\bw2color_image.py --image D:\coding\python\bw-colorization\images\IMG_20200301_011028.jpg --prototxt D:\coding\python\bw-colorization\model\colorization_deploy_v2.prototxt --model  D:\coding\python\bw-colorization\model\colorization_release_v2.caffemodel --points D:\coding\python\bw-colorization\model\pts_in_hull.npy

