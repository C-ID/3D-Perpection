#!
function init()
{
	binfile='/home/bai/Project/3D-Perpection/3d-sample/bin_files/'
	result_dir='/home/bai/Project/3D-Perpection/result/'
	outfile='/home/bai/Project/3D-Perpection/per/cmake-build-debug/libs/'
	pythonpath='/home/bai/Project/3D-Perpection/vtk-test/'
	bagpath='/home/bai/pcd/merge/'
	suffix='.json'
	png='/home/bai/pcd/png/merge/'
	json='/home/bai/pcd/json/cent/'
	pngsuffix='.png'
}

function check_result_path()
{
	if [ ! -d result_dir ]; then
		mkdir -p $result_dir
	else
		echo "all result json files will be saved to ${result_dir}"
	fi
}
	
function dump()
{
	cd ${outfile}
	i=0
	for file in `ls $binfile`;
	do
		./out ${binfile}${file} ${result_dir}${i}${suffix};	
		i=$((i+1))
	done
	echo "dump 100 cloud point cnn seg results to json"
}

function render()
{
	i=0
	source activate vtk 
	source ~/.zshrc
	for file in `ls $binfile`;
	do
		echo "json file: "${i}${suffix} "bin file: "${file}
		python ${pythonpath}render.py --json-path ${result_dir}${i}${suffix} \
					 				     --pcd-path ${binfile}${file}

		i=$((i+1))
	done
	echo "render finished"
}

function bag()
{
	cd ${outfile}
	
	i=0
	for bag in `ls $bagpath`; 
	do
		./out ${bagpath}${bag} ${json}${i}${suffix} ${png}${i}${pngsuffix};
		i=$((i+1))
	done
	echo "dump bag file cnn seg result to json finished"
}


function main()
{
	init
	#check_result_path
	#dump
	local cmd=$1
  	
  	case $cmd in
  		dump)
		dump $@
		;;
		render)
		render $@
		;;
		both)
		dump $@
		render $@
		;;
		bag)
		bag $@
		;;
	esac
}

main $@
