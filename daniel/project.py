import numpy as np, imageio, scipy.io, sys, os, cv2
# from PIL import Image
import matplotlib.pyplot as plt

def define_psf(U, V, slope):
	# Vars
	x = np.linspace(-1, 1, 2 * U)
	y = np.linspace(-1, 1, 2 * U)
	z = np.linspace(0, 2, 2 * V)
	gz, gy, gx = [np.swapaxes(i, 0, 1) for i in np.meshgrid(z, y, x)]
	print(f'x, y, z shape: {x.shape}, {y.shape}, {z.shape}')
	print(f'gx, gy, gz shape: {gx.shape}, {gy.shape}, {gz.shape}')

	# Define PSF
	psf = np.abs((4 * slope)**2 * (gx**2 + gy**2) - gz).astype(float)
	print(f'psf shape: {psf.shape}')
	# psf = psf == repmat(min(psf,[],1),[2.*V 1 1]);
	# psf_min = np.expand_dims(psf.min(0), 0)
	# psf_min = psf.min(0)
	# print(f'psf_min shape: {psf_min.shape}')
	psf = np.tile(psf.min(0), [2 * V, 1, 1])
	print(f'psf tiled shape: {psf.shape}')
	psf = psf / psf[:,U,U].sum()
	psf = psf / np.linalg.norm(psf);
	# psf_norm = np.linalg.norm(psf)
	# print(f'psf norm: {psf_norm}')
	# psf = circshift(psf,[0 U U])
	psf = np.roll(psf, (0, U, U))
	
	return psf

def resampling_operator(M):
	# Vars
	# mtx = np.zeros((M**2, M))
	# mtx = np.ones((M**2, M))
	# print(f'mtx shape: {mtx.shape}')
	# x = np.asarray([i for i in range(M**2)])
	# xs = np.floor(x**0.5)
	# print(x.min(), x.max())
	# print(xs.min(), xs.max())
	# mtx_ind = np.ravel_multi_index((x, xs), mtx.shape)
	# print(f'mtx_ind shape: {mtx_ind.shape}')
	# print(mtx_ind[0:10])
	# mtx[mtx_ind] = 1
	# print(f'mtx shape: {mtx.shape}')

	# mtx_diag = spdiags(1./sqrt(x)',0,M.^2,M.^2)
	mtx = np.zeros((M**2, M))
	x = np.asarray([i for i in range(M**2)])
	xs = np.floor(x**0.5).astype(int)
	print(f'xs zeros: {len([i for i in xs if i == 0])}')
	mtx[x, xs] = 1
	# print(mtx.min(), mtx.max(), mtx.sum(), mtx.shape)

	xs[xs == 0] = 1
	mtx = scipy.sparse.csr_matrix(mtx)
	# mtx_diag = scipy.sparse.spdiags(1 / xs, 0, M**2, M**2)
	mtx = scipy.sparse.spdiags(1 / xs, 0, M**2, M**2) @ mtx 
	mtxi = mtx.T
	print(f'mtx shape: {mtx.shape}')
	print(f'mtxi shape: {mtxi.shape}')

	K = np.round(np.log(M) / np.log(2)).astype(int)
	print(f'K: {K}')
	for i in range(K):
		mtx = 0.5 * (mtx[::2, :] + mtx[1::2, :])
		mtxi = 0.5 * (mtxi[:, ::2] + mtxi[:, 1::2])

	return mtx.toarray(), mtxi.toarray()

def cnlos_reconstruction(mat_in_path):
	##### SETUP DATA #####
	# Constants
	isbackprop = False
	isdiffuse = False
	bin_resolution = 4e-12
	c = 3e8
	K = 2
	snr = 8e-1
	z_trim = 600
	z_offset_dict = {
		'data_resolution_chart_40cm.mat' : [350],
		'data_resolution_chart_65cm.mat' : [700],
		'data_dot_chart_40cm.mat' : [350],
		'data_dot_chart_65cm.mat' : [700],
		'data_mannequin.mat' : [300],
		'data_exit_sign.mat' : [600],
		'data_s_u.mat' : [800],
		'data_outdoor_s.mat' : [700],
		'data_diffuse_s.mat' : [100, 1, snr * 1e-1],
	}

	# Open matlab data file
	mat = scipy.io.loadmat(mat_in_path)
	mat_width, mat_rect_data = mat['width'][0, 0], mat['rect_data']
	print(f'mat_width: {mat_width}')
	print(f'mat_rect_data shape: {mat_rect_data.shape}')

	# Get fixed z offset and other constant params
	mat_fname = os.path.split(mat_in_path)[-1]
	print(f'mat_fname: {mat_fname}')
	mat_params = z_offset_dict[mat_fname]
	if len(mat_params) > 1: z_offset, isdiffuse, snr = mat_params
	else: z_offset = mat_params[0]
	print(f'z_offset, isdiffuse, snr: {z_offset}, {isdiffuse}, {snr}')

	# Get spatial, temporal, and range dims
	N, M = mat_rect_data.shape[0], mat_rect_data.shape[-1]
	mat_range = M * c * bin_resolution
	print(f'M, N: {M}, {N}')

	# Downsample data to 16 picoseconds
	for i in range(K):
		M //= 2
		bin_resolution *= 2
		mat_rect_data = mat_rect_data[:, :, 0::2] + mat_rect_data[:, :, 1::2]
		z_trim = round(z_trim / 2)
		z_offset = round(z_offset / 2)
	print(f'M, N: {M}, {N}')

	# Set first group of histogram bins to zero (to remove direct component)
	mat_rect_data[:, : , 0:z_trim] = 0

	# Define NLOS blur kernel 
	psf = define_psf(N, M, mat_width / mat_range);
	print(f'psf shape: {psf.shape}')
	scipy.io.savemat('psf_my.mat', {'psf_mine' : psf})

	# Compute inverse filter of NLOS blur kernel
	fpsf = np.fft.fftn(psf)
	if isbackprop: invpsf = np.conj(fpsf)
	else: invpsf = np.conj(fpsf) / (np.abs(fpsf)**2 + 1.0 / snr)

	# Define volume representing voxel distance from wall
	Ml = np.linspace(0, 1, M).T
	grid_z = np.tile(Ml[:, None, None], [1, N, N])
	print(f'grid_z shape: {grid_z.shape}')

	# Get transform operators and permute data dims
	mtx, mtxi = resampling_operator(M)
	data = np.asarray(mat_rect_data.transpose(2, 1, 0))


	##### RUN ALGO #####
	# Step 1: Scale radiometric component
	if isdiffuse: data = data * grid_z**4
	else: data = data * grid_z**2
	print(f'data shape: {data.shape}')
	# print(data[-1, 0:10, 0])

	# Step 2: Resample time axis and pad result
	# tdata = zeros(2.*M,2.*N,2.*N);
	# tdata(1:end./2,1:end./2,1:end./2)  = reshape(mtx*data(:,:),[M N N]);
	data_rs = data.reshape((data.shape[0], -1))
	print(f'data_rs shape: {data_rs.shape}')
	# print(data_rs[-1, 0:10])
	tdata = np.zeros((2 * M, 2 * N, 2 * N))
	tdata[:M, :N, :N] = (mtx @ data_rs).reshape((M, N, N))
	print(f'tdata shape: {tdata.shape}')


	# Step 3: Convolve with inverse filter and unpad result
	# tvol = ifftn(fftn(tdata).*invpsf);
	# tvol = tvol(1:end./2,1:end./2,1:end./2);
	tvol = np.fft.ifftn(np.fft.fftn(tdata) * invpsf)
	tvol = tvol[:M, :N, :N]
	print(f'tvol shape: {tvol.shape}')

	# Step 4: Resample depth axis and clamp results
	# vol  = reshape(mtxi*tvol(:,:),[M N N]);
	# vol  = max(real(vol),0);
	tvol_rs = tvol.reshape((tvol.shape[0], -1))
	vol = (mtxi @ tvol_rs).reshape((M, N, N))
	# print(f'vol shape: {vol.shape}')
	vol = np.real(vol)
	vol[vol < 0] = 0
	print(f'vol shape: {vol.shape}')

	tic_z = np.linspace(0, mat_range//2, vol.shape[0])
	tic_y = np.linspace(-mat_width, mat_width, vol.shape[1])
	tic_x = np.linspace(-mat_width, mat_width, vol.shape[2])
	
	# Crop and flip for visualization
	# ind = round(M.*2.*width./(range./2));
	# vol = vol(:,:,end:-1:1);
	# vol = vol((1:ind)+z_offset,:,:);
	ind = round(M * 2 * mat_width / (mat_range / 2))
	print(f'ind: {ind}, {M}, {mat_width}, {mat_range}, {z_offset}')
	vol = vol[:, :, -1::-1]
	vol = vol[z_offset:z_offset+ind, :, :]
	print(f'vol shape: {vol.shape}')
	vol1 = vol.max(0)
	print(f'vol1 shape: {vol1.shape}')
	scipy.io.savemat('vol1_my.mat', {'vol1_mine' : vol1})

	tic_z = tic_z[z_offset:ind+z_offset]
	
	# plt.figure('bob')

	# plt.subplot(1, 3, 1)
	# plt.imshow(vol1, aspect='auto', interpolation='none', extent=None, cmap='gray')
	# plt.show()

def main():
	# Args
	try:
		mat_in_path = sys.argv[1]
		img_out_path = sys.argv[2]
	except:
		print(f'\n***ERROR*** Must have two positional arguments for path to matlab file and output file:\n\n$ python3 project.py path/to/data.m path/to/output.png\n')
		sys.exit()

	#CNLOS Reconstruction
	rec = cnlos_reconstruction(mat_in_path)

if __name__ == '__main__':
	main()