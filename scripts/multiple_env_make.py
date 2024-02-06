from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np
from terrain_utils_update import *
import math
from scipy import interpolate
import copy
from transforms3d.euler import quat2euler
from scipy.spatial.transform import Rotation as R
import os
import cv2
import core
from torchvision import transforms

class env():
    def __init__(self,tracked_dofs_pos=[],tracked_dofs_vel=[],tracked_root=[],num_groups=10, initial_height = 1.0,viewer_flag=False):
        
        self.tracked_dofs_pos=tracked_dofs_pos  # Dof's to track with costs
        self.tracked_dofs_vel=tracked_dofs_vel  # Dof's to track with costs
        self.tracked_root=tracked_root  # Root Dof's to track with costs
        self.viewer_flag = viewer_flag
        self.num_groups = num_groups
        self.group_assignments = np.random.randint(0, num_groups, size=2**8)
        self.ledge_heights = [initial_height] * num_groups


    def _setup_env(self,asset_file='robot_mvw.urdf'):
       # initialize gym
        gym = gymapi.acquire_gym()

        # parse arguments
        args = gymutil.parse_arguments(description="Multiple Cameras Example",
                               custom_parameters=[
                                   {"name": "--save_images", "action": "store_true", "help": "Write RGB and Depth Images To Disk"},
                                   {"name": "--up_axis_z", "action": "store_true", "help": ""}])
        args.save_images = True
        self.args = args

        sim_parms = gymapi.SimParams()
        sim_parms.up_axis = gymapi.UpAxis.UP_AXIS_Z
        sim_parms.gravity = gymapi.Vec3(0.0, 0.0, -9.81)        
        sim_parms.dt = 0.02
        # gymapi.FlexParams.static_friction=1.0'robot_mvw.urdf'
        # gymapi.FlexParams.deterministic_mode=True 
        sim_parms.physx.num_velocity_iterations=5
        sim_parms.physx.num_position_iterations=10
        # sim_parms.physx.contact_offset=0.04
        # sim_parms.physx.bounce_threshold_velocity=0.5
        sim_parms.physx.use_gpu = True
        sim_parms.use_gpu_pipeline = True
        if sim_parms.use_gpu_pipeline == True:
            self.device='cuda:0'
        else:
            self.device='cpu'
        sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_parms)
        self.sim = sim

        # add ground plane
        # **************** NEW ADDITION MIGHT NEED TO REMOVE *********************
        # plane_params = gymapi.PlaneParams()
        # plane_params.normal.y=0
        # plane_params.normal.z=1.
        # # plane_params.distance = -1.95
        # gym.add_ground(sim, plane_params)

        # ************************************************************************

        # create viewer
        if self.viewer_flag == True:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                print("*** Failed to create viewer")
                quit()
            # position the camera
            cam_pos = gymapi.Vec3(17.2, 10.0, 16)
            cam_target = gymapi.Vec3(0, -10.5, -5)
            gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
            self.viewer = viewer

        # load asset
        asset_root = "../assets/"
        # asset_file = 'robot2.urdf'

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link =False
        asset_options.use_mesh_materials = False

        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        # Camera
        camera_prop = gymapi.CameraProperties()
        camera_prop.horizontal_fov = 75.0
        camera_prop.width = 1920
        camera_prop.height = 1080

        

        # get array of DOF names
        dof_names = gym.get_asset_dof_names(asset)

        # get array of DOF properties
        dof_props = gym.get_asset_dof_properties(asset)

        # create an array of DOF states that will be used to update the actors
        num_dofs = gym.get_asset_dof_count(asset)
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

        # get list of DOF types
        dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

        # get the position slice of the DOF state array
        dof_positions = dof_states['pos']

        # set up the env grid
        num_envs = int(2**1)
        self.num_envs = num_envs
        actors_per_env = 1
        dofs_per_actor = 11
        num_per_row = 3
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # cache useful handles
        self.envs = []
        self.actor_handles = []

        # Hidden state inialization for GRU

        self.hidden_state_layers = 64
        # self.hidden_state = torch.zeros(1, self.num_envs, 64).to(device='cuda')
        self.hidden_state = torch.zeros(1, self.num_envs, 64).to(device='cuda:0')

        print("Creating %d environments" % num_envs)
        self.env_origin=[]
        # track cameras
        i_map = [i for i in range(num_envs)]
        self.cam_arr = {idx: {} for idx in i_map}
        # ch_map = {idx:{} for idx in env_idxs}
        pos = gymapi.Vec3(0,0,0)
        # camera = gym.create_camera_sensor(name='camera', width=640, height=480, fov=60, near=0.1, far=100)
        # camera_pose = ([0,0,0],[0,0,0])
        # gym.attach_camera_to_robotIs 
        for i in range(num_envs):
            # create env
            self.env = gym.create_env(sim, env_lower, env_upper, num_per_row)
            self.envs.append(self.env)
            name = f"cam{i}"
            cam = gym.create_camera_sensor(self.env, camera_prop)
            self.cam_arr[i][name] = cam
            gym.set_camera_transform(cam, self.env, gymapi.Transform(p=pos, r=gymapi.Quat()))
            # print("Added {} with handle {} in env {} | View matrix: {}".format(name, cam, i, gym.get_camera_view_matrix(sim, self.env, cam)))

            # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 1.32, .0)
            pose.r = gymapi.Quat(0, 0.0, 0., 1.0)
            # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

            self.actor_handle = gym.create_actor(self.env, asset, pose, "actor", i, 1)
            self.actor_handles.append(self.actor_handle)
            # props = gym.get_actor_dof_properties(self.env, self.actor_handle)
            # props['driveMode'].fill(gymapi.DOF_MODE_VEL)
            # props['stiffness'].fill(10.)
            # props['damping'].fill(2.)
            # props['driveMode'][-1]=gymapi.DOF_MODE_VEL
            # props['stiffness'][-1]=1000.
            # props['damping'][-1]=200.
            # targets = np.zeros(num_dofs).astype('f')
            # gym.set_actor_dof_position_targets(self.env, self.actor_handle, targets)
            # vel_targets = 10*np.ones(4).astype('f')#np.random.uniform(-math.pi, math.pi, 4).astype('f')
            # gym.set_actor_dof_velocity_targets(self.env, self.actor_handle, vel_targets)
            # set default DOF positions
            gym.set_actor_dof_states(self.env, self.actor_handle, dof_states, gymapi.STATE_ALL)
            self.env_origin.append([gym.get_env_origin(self.envs[i]).x,gym.get_env_origin(self.envs[i]).y,gym.get_env_origin(self.envs[i]).z])
        
        if args.save_images and not os.path.exists("camera_images"):
            os.mkdir("camera_images")
        gym.render_all_camera_sensors(sim)
        self.env_origin=torch.tensor(np.array(self.env_origin)).cuda()
        # self.env_origin=torch.tensor(np.array(self.env_origin),device="cpu",dtype=torch.float32)
        # create all available terrain types
        num_terains = 2
        terrain_width = 55.
        terrain_length = 100.*5
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.03  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)


        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)


        # heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.2, step=0.2, downsampled_scale=0.5).height_field_raw
        # heightfield[num_rows:2*num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
        # heightfield[2*num_rows:3*num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
        # heightfield[:, :] = discrete_obstacles_terrain(new_sub_terrain(), max_height=10.5, min_size=10., max_size=5., num_rects=2000).height_field_raw
        # heightfield[4*num_rows:5*num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=2., amplitude=1.).height_field_raw
        # heightfield[:, :] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
        # heightfield[6*num_rows:7*num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
        # heightfield[7*num_rows:8*num_rows, :] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.,
        #                                                         stone_distance=1., max_height=0.5, platform_size=0.).height_field_raw

        # heightfield=step_terrain(new_sub_terrain())
        heightfield = sand_dune_terrain(new_sub_terrain())
        # heightfield=discrete_obstacles_terrain(new_sub_terrain())
        # heightfield == pyramid_sloped_terrain(new_sub_terrain())
        # heightfield=random_uniform_terrain(new_sub_terrain())
        # heightfield = wave_terrain(new_sub_terrain())

        # add the terrain as a triangle mesh
        # heightfield=np.zeros((400,400),dtype='int16')
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        self.tt=torch.tensor((np.copy(vertices.reshape((int(terrain_width/horizontal_scale),int(terrain_length/horizontal_scale),3))))).cuda()

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        # tm_params.transform.p.x = -1.
        # tm_params.transform.p.y = -1.
        # tm_params.transform.r.w=math.cos(math.pi/4)
        # tm_params.transform.r.x=math.cos(math.pi/4)
        off_x=-20.
        off_y=-5.
        off_z=-1
      
        tm_params.transform.p.x = off_x
        tm_params.transform.p.y = off_y
        tm_params.transform.p.z = off_z   
        self.tt[:,:,0]=self.tt[:,:,0]+off_x  
        self.tt[:,:,1]=self.tt[:,:,1]+off_y 
        self.tt[:,:,2]=self.tt[:,:,2]+off_z
        gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

        # for i in range(num_envs):
        #     name = f"cam{i}"
        #     env = self.envs[i]
        #     cam = self.cam_arr[i][name]
        #     rgb_filename = "camera_images/rgb_env%d_cam0.png" % (i)
        #     if args.save_images:
        #         gym.write_camera_image_to_file(sim,env, cam, gymapi.IMAGE_COLOR, rgb_filename)
        #     print("View matrix of {} with handle {} in env {}: {}".format(name, cam, i, gym.get_camera_view_matrix(sim, env, cam)))


        gym.prepare_sim(sim)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)    

        # ## Aquire tensor descriptors #
        self.root_states_desc = gym.acquire_actor_root_state_tensor(sim)
        self.dof_states_desc = gym.acquire_dof_state_tensor(sim)
        self.force_buff=gym.acquire_dof_force_tensor(sim)

        # ## Pytorch interop ##
        root_states = gymtorch.wrap_tensor(self.root_states_desc)
        # row = torch.tensor([[0,0,0.3746,0.9238]], device='cuda:0')
        # root_states[:,3:7] = row.repeat(self.num_envs,1)
        # state = gym.get_actor_rigid_body_states
        dof_states = gymtorch.wrap_tensor(self.dof_states_desc)
        self.dof_force=gymtorch.wrap_tensor(self.force_buff).reshape(num_envs,num_dofs)
        additional_tensor = torch.zeros_like(root_states)
        additional_tensor[:,1] = 1
        self.flag1 = np.zeros(num_envs)
        
        self.prev_root_state = torch.clone(root_states)
        self.new_position = torch.zeros_like(self.prev_root_state)
        self.prev_dof_state = torch.clone(dof_states)

        # ## View information as a vector of envs ##
        self.root_states_vec = root_states.view(num_envs,1,13)

        self.shift = additional_tensor
        self.dof_states_vec = dof_states.view(num_envs,int(dof_states.size()[0]/num_envs),2)
        
        # self.rec_state=torch.zeros((num_envs,int(dof_states.size()[0]/num_envs),self.T),dtype=torch.float32, device=self.device)
        # self.new_state=torch.zeros((int(dof_states.size()[0]/num_envs),self.T),dtype=torch.float32, device=self.device)

        self.gym = gym
        self.sim = sim
        self.num_envs=num_envs
        self.num_dofs = num_dofs
        self.weights=torch.zeros(self.num_envs,len(self.tracked_root)+len(self.tracked_dofs_vel)+len(self.tracked_dofs_pos),len(self.tracked_root)+len(self.tracked_dofs_pos)+len(self.tracked_dofs_vel)).to(self.device)
        self.des_state=torch.zeros((self.num_envs,len(self.tracked_root)+len(self.tracked_dofs_vel)+len(self.tracked_dofs_pos),1),device="cuda:0",dtype=torch.float32)
        self.des_state[:,2]=3.0
        weight_diag=torch.tensor([0.,-0.001,5.,-0.001,0,0,0,0,0,0],device="cuda:0",dtype=torch.float)
        self.weights[:]=torch.diag(weight_diag)
        self.dof_forces = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)).view(self.num_envs,-1)
        # self.weights[:]=torch.diag(torch.tensor([-0.0,-0.1,-.1,1.0,-0.],dtype=torch.float))
        # self.weights[:]=torch.diag(torch.tensor([-0.01,-0.01,-.01,1.0,-0.001,-0.001,-0.001,-0.001,-0.01],dtype=torch.float))
        # self.weights[:]=torch.diag(torch.tensor([-0.5,1.0,-0.01,-0.1],dtype=torch.float))
        # self.weights[:]=torch.diag(torch.tensor([-0.5,1.0,-0.00,-0.0],dtype=torch.float))


        # Setup scandot stuff
        # self.wb_xpts=torch.linspace(0.,1.,5,device="cuda:0")
        # self.wb_ypts=torch.linspace(-1.,1.,5,device="cuda:0")
        width=5
        length=5
        step_length=2
        step_width=2
        self.scan_dot_buf=torch.zeros((num_envs,width,length,3),device="cuda:0",dtype=torch.float32)        
        self.wb_state=self.root_states_vec[:,0,:3]+self.env_origin
        self.wb_ind=torch.zeros((num_envs,2),device="cuda:0",dtype=torch.int16)
        self.tt = self.tt.contiguous()
        
        for i in range(num_envs):

            slice_contigous1 = self.tt[:,0,0].contiguous()
            slice_contigous2 = self.tt[0,:,1].contiguous()
            self.wb_ind[i,0]=torch.searchsorted(slice_contigous1, self.wb_state[i,0]).item()
            self.wb_ind[i,1]=torch.searchsorted(slice_contigous2, self.wb_state[i,1]).item()
        # This device will remain cpu
        self.wb_ind=self.wb_ind.to(device="cpu").detach().numpy()
        stride_length=step_length/horizontal_scale/(length-1)
        stride_width=step_width/horizontal_scale/(width-1)
        self.stride_ind=[]
        for i in range(width):
            for j in range(length):
                self.stride_ind.append([int(i*stride_width-step_length/(2*horizontal_scale)),int(j*stride_length-step_width/(2*horizontal_scale))])
        self.stride_ind=np.array(self.stride_ind)
        self.scan_dot_buf=torch.zeros((num_envs,width*length,3),device="cuda:0",dtype=torch.float32)
        self.safe_check = 0
        self.step()
        self.reset()       
        
        # y_ind=torch.searchsorted(self.tt[0,:,1], x[1]+ypts).item()
        # self.observation_space = torch.hstack((self.tracked_states_vec,self.gamma.reshape(self.num_envs,-1,1)))#,self.image.reshape(self.num_envs,-1,1)))
        self.observation_space = torch.hstack((self.tracked_states_vec,self.scan_dot_buf.reshape(self.num_envs,-1,1)))
        self.action_space = torch.zeros((self.num_envs,5),dtype=torch.float,device=self.device) 
        self.prev_vel = torch.zeros((self.num_envs,4),dtype=torch.float,device=self.device) 
        self.prev_action = torch.zeros((self.num_envs,5),dtype=torch.float,device=self.device) 

    def _render(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)

    def _compute_reward(self):
        # pendulum position is 3
        rew_speed=10.*torch.exp(-(self.tracked_states_vec[:,2]-3.)**2).cuda()
        rew_torque=-0.001*torch.norm(self.dof_forces[:,:4],dim=1).unsqueeze(dim=1).cuda()-0.0001*torch.abs(self.tracked_states_vec[:,-1]).cuda()
        # rew_torque = rew_torque.to(device='cuda')
        # rew_pos=-0.01*self.tracked_states_vec[:,1]**2
        rew_pos = (10*self.position[:,0,0]**2).unsqueeze(1).cuda()
        rew_ori = (torch.tensor(-0.01*np.absolute(self.orientation[:,2])**2).unsqueeze(1)).cuda()
        # rew_ori = (torch.tensor(-0.01*np.absolute(self.orientation[:,2])**2).unsqueeze(1))
        rew_flag = (torch.tensor(100*self.flag1).unsqueeze(1)).cuda()
        # rew_flag = (torch.tensor(100*self.flag1).unsqueeze(1))
        return rew_speed+rew_torque+rew_pos+rew_ori+rew_flag#torch.sum(torch.bmm(self.weights,(self.tracked_states_vec-self.des_state)),dim=1)#-0.1*torch.norm(self.dof_force[:,:4],dim=1).reshape((self.dof_force.size()[0]),1)
        
    def _terminal_flag(self):
        # return torch.where(self.tracked_states_vec[:,3,0]<0.5)[0]
        # if (self.tracked_states_vec[:,0,0]<0.65)[0]:
        #     print("s")
        return torch.where(self.tracked_states_vec[:,0,0]<0.65)[0]
        # return torch.where(self.tracked_states_vec[:,0,0]<0.707)[0]
        # print('teop')
        # if self.tracked_states_vec[:,3,0]:
        #     return True
        # else:
        #     return False
    def pd_ctrl(self,des_action):
        kpv=7.
        kdv=0.5
        kpp=2000.
        kdp=200.
        # des_action.clip(min=-5.,max=5.)
        # action=1*des_action*torch.ones((des_action.size()[0],4),device="cuda:0",dtype=torch.float32)
        # des_action=des_action*torch.ones((des_action.size()[0],5),device="cuda:0",dtype=torch.float32)

        des_action[:,:-1]=kpv*(des_action[:,:-1]-self.dof_states_vec[:,:4,1])-kdv*(self.dof_states_vec[:,:4,1]-self.prev_vel)
        des_action[:,-1]=kpp*(des_action[:,-1]-self.dof_states_vec[:,-1,0])-kdp*(self.dof_states_vec[:,-1,1])

        # action=kp*(des_action*torch.ones((des_action.size()[0],4),device="cuda:0",dtype=torch.float32)-self.dof_states_vec[:,:4,1])-kd*(self.dof_states_vec[:,:4,1]-self.prev_vel)
        # action=kp*(des_action-self.dof_states_vec[:,:4,1])-kd*(self.dof_states_vec[:,:4,1]-self.prev_vel)
        self.prev_vel=torch.clone(self.dof_states_vec[:,:4,1])
        return des_action

    def step(self,action=[]):
        # forces_desc = gymtorch.unwrap_tensor((self.U[:,:,i]+self.delta_U[:,:,i]).contiguous())
        
        if len(action)>0:
            action_out=self.pd_ctrl(torch.clone(action))
            self.dof_force[:,:]=action_out
            forces_desc = gymtorch.unwrap_tensor(self.dof_force)
            self.gym.set_dof_actuation_force_tensor(self.sim,forces_desc)
            # cv2.imshow(image)
            # rgb_filename = "camera_images/rgb_env%d_cam0.png" % (i)
            # if self.args.save_images:
            #     self.gym.write_camera_image_to_file(self.sim,env, cam, gymapi.IMAGE_COLOR, rgb_filename)
            # print("View matrix of {} with handle {} in env {}: {}".format(name, cam, i, self.gym.get_camera_view_matrix(self.sim, env, cam)))


        # image = self.image_cnn()
        # image_reshape = image.reshape(self.num_envs,-1,1)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.render_all_camera_sensors(self.sim)
        self._refresh_state()
        # self.gamma = self.gru_network1(self.scan_dot_buf)
        # z = core.GRUNetwork2(torch.cat((self.tracked_states_vec,self.scan_dot_buf.reshape(self.num_envs,-1,1)), dim=1))
        try:
            # next_obs=torch.hstack((self.tracked_states_vec,self.gamma.reshape(self.num_envs,-1,1)))#,image.reshape(self.num_envs,-1,1)))
            next_obs = torch.hstack((self.tracked_states_vec,self.scan_dot_buf.reshape(self.num_envs,-1,1)))
        except:
            # next_obs=torch.hstack((self.tracked_states_vec,torch.zeros_like(self.gamma.reshape(self.num_envs,-1,1))))#,torch.zeros_like(image.reshape(self.num_envs,-1,1))))
            next_obs = torch.hstack(self.tracked_states_vec,torch.zeros_like(self.scan_dot_buf.reshape(self.num_envs,-1,1)))
        reward = self._compute_reward()
        done = self._terminal_flag()
        info=[]

        if self.viewer_flag==True:
            self._render()
        return next_obs, reward, done, info, self.scan_dot_buf#self.dof_states_vec[0,:4,1].to("cpu").detach().numpy(), self.dof_states_vec[0,-1,0].to("cpu").detach().numpy()

    def gru_network1(self, input):
        self.gru1 = core.GRUNetwork1(input_size=3, hidden_size=self.hidden_state_layers, gru_layers=1)#.to(device='cuda')
        input = input.permute(1,0,2)
        gamma, hidden_state = self.gru1(input,self.hidden_state)
        self.hidden_state = hidden_state
        gamma = gamma.permute(1,0,2)
        return gamma
    def image_cnn(self):
        image_input_height = 256
        image_input_width = 256
        self.image = torch.zeros((self.num_envs,image_input_width,image_input_height,3),device="cuda:0",dtype=torch.float32)

        for i in range(self.num_envs):
            name = f"cam{i}"
            env = self.envs[i]
            cam = self.cam_arr[i][name]
            image = self.gym.get_camera_image(self.sim,env,cam,gymapi.IMAGE_COLOR)
            image = image.reshape(image.shape[0],-1,4)[...,:3]
            aspect_ratio = image.shape[1]/image.shape[0]
            # new_height = int(new_width/aspect_ratio)
            resized_image = cv2.resize(image, (image_input_width,image_input_height), interpolation=cv2.INTER_AREA)
            resized_image = torch.from_numpy(resized_image).type(torch.float32).to(device = "cuda:0")
            self.image[i,:,:] = resized_image
        self.image = self.image.permute(0,3,1,2)
        cnn_extractor = core.CNNFeatureExtractor().to('cuda')
        image = cnn_extractor(self.image)
        return image        


    def flag(self, idx=[]):
        # flag = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            if i%3 == 0:
                if self.root_states_vec[i,0,0] > 10.0:
                    print(f"{i}, went over")
                    if self.root_states_vec[i,0,1] > 100.:
                        print("Going in 3rd")
                    self.flag1[i] = 1
            elif (i+1)%3 == 0:
                if self.root_states_vec[i,0,0] > 8.0:
                    print(f"{i}, went over")
                    if self.root_states_vec[i,0,1] > 100.:
                        print("Going in 3rd")
                    self.flag1[i] = 1
            elif (i+2)%3 == 0:
                if self.root_states_vec[i,0,0] > 6.5:
                    print(f"{i}, went over")
                    if self.root_states_vec[i,0,1] > 100.:
                        print("Going in 3rd")
                    self.flag1[i] = 1
        # self.flag1 = flag
                


        return self.flag1
    
    def reset(self,idx=[]):
        ## Need to randomize base ##
        if len(idx)<1:
            idx=np.arange(0,self.num_envs)

        current_root_state=torch.clone(self.root_states_vec[:,0,:])
        # row = torch.tensor([[0,0,0.7071,0.7071]], device='cuda:0')
        # current_root_state[:,3:7] = row.repeat(256,1)

        current_dof_state=torch.clone(self.dof_states_vec)
        for i in idx:

            if self.safe_check > 1:                                                                                                                                                                                                                                                                        

                shift_arr = self.flag1[i]*self.shift[i,1]*100
                # if self.flag1[i] == 1:
                #     self.shift[i,1] = 100
                # self.new_position = self.prev_root_state
                if self.prev_root_state[i,1] > 400:
                    self.prev_root_state[i,1] = np.random.randint(25,475)
                else:
                    self.prev_root_state[i,1] = self.prev_root_state[i,1] + shift_arr

                # self.new_position[i,1] = self.prev_root_state[i,1] + shift_arr
            if self.prev_root_state[i,1] > 200:
                self.prev_root_state[i,2] = -2.5
            if self.prev_root_state[i,1] > 400:
                self.prev_root_state[i,2] = -3.5

            
            current_root_state[i,:]=torch.clone(self.prev_root_state[i])

            if self.flag1[i] == 1:
                self.flag1[i] = 0 
            
            # current_dof_state[i,:,:]=torch.clone(self.prev_dof_state.view(self.num_envs,self.num_dofs,2)[i]+3.14)
            # current_dof_state[i,:,:]=torch.clone(self.prev_dof_state.view(self.num_envs,self.num_dofs,2)[i]-1.5*(torch.rand((self.num_dofs,2),dtype=torch.float,device=self.device)-0.5))
            # current_dof_state[i,-1,0]=3.14+0.5*(torch.rand((1,1),dtype=torch.float,device=self.device)-0.5)
            # current_dof_state[i,-1,1]=1.*(torch.rand((1,1),dtype=torch.float,device=self.device)-0.5)
            # current_dof_state[i,-1,0]=3.14*(torch.rand((1,1),dtype=torch.float,device=self.device)-0.5)
        self.safe_check += 1
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(current_root_state))
        self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(current_dof_state))       
        # self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.prev_root_state.view(self.num_envs,13)))
        # self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.prev_dof_state.view(self.num_envs,self.num_dofs,2)+0.5*torch.rand((self.num_envs,self.num_dofs,2),dtype=torch.float,device=self.device)))
        self._refresh_state()
        # next_obs=torch.hstack((self.tracked_states_vec,self.gamma.reshape(self.num_envs,-1,1)))#,self.image.reshape(self.num_envs,-1,1)))#self.tracked_states_vec
        next_obs = torch.hstack((self.tracked_states_vec,self.scan_dot_buf.reshape(self.num_envs,-1,1)))
        return next_obs
    
    def _refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.tracked_states_vec=torch.cat((torch.abs(self.root_states_vec[:,0,self.tracked_root[:2]]),self.root_states_vec[:,0,self.tracked_root[2:]],self.dof_states_vec[:,self.tracked_dofs_pos,0],torch.abs(self.dof_states_vec[:,self.tracked_dofs_vel,1])),1).view(self.num_envs,len(self.tracked_dofs_vel)+len(self.tracked_dofs_pos)+len(self.tracked_root),1)
        # self.state = self.gym.get_actor_rigid_body_states(self.sim, self.actor_handle, gymapi.STATE_ALL)
        self.position = self.root_states_vec[:,:3]
        self.orientation_quat = self.root_states_vec[:,:,3:7].reshape(self.num_envs, 4)
        # rotation = R.from_quat(orientation_quat)
        rotation = R.from_quat(self.orientation_quat.cpu().numpy())
        self.euler_angles = rotation.as_euler('xyz', degrees = False)
        self.orientation = np.degrees(self.euler_angles)
        # row = torch.tensor([[0,0,0.7071,0.7071]], device='cuda:0')
        # self.root_states_vec[:,0,3:7] = row.repeat(256,1)
        # for i in range(self.num_envs):
        #     self.root_states_vec[i,:,3:7] = torch.tensor([[0,0,0.7071,0.7071]], device='cuda:0')
        # self.tracked_states_vec=torch.cat((torch.abs(self.root_states_vec[:,0,self.tracked_root]),self.dof_states_vec[:,self.tracked_dofs_pos,0],torch.abs(self.dof_states_vec[:,self.tracked_dofs_vel,1])),1).view(self.num_envs,len(self.tracked_dofs_vel)+len(self.tracked_dofs_pos)+len(self.tracked_root),1)
        # self.tracked_states_vec[:,3,0]=1+torch.cos(self.tracked_states_vec[:,3,0])
        # self.tracked_states_vec[:,1,0]=1+torch.cos(self.tracked_states_vec[:,1,0])
        # self.get_surrounding_terrain()
        # self.tracked_states_vec[0,0,0] = 0.65
        self.update_scandots()

    def update_scandots(self):
        # update wb index
        self.wb_state=self.root_states_vec[:,0,:3]+self.env_origin

        for i in range(self.num_envs):
            try:
                self.wb_ind[i,0]+=torch.searchsorted(self.tt[self.wb_ind[i,0]-3:self.wb_ind[i,0]+3,0,0], self.wb_state[i,0]).item()-3
                self.wb_ind[i,1]+=torch.searchsorted(self.tt[0,self.wb_ind[i,1]-3:self.wb_ind[i,1]+3,1], self.wb_state[i,1]).item()-3
                self.scan_dot_buf[i,:,:]=self.tt[self.wb_ind[i,0]+self.stride_ind[:,0],self.wb_ind[i,1]+self.stride_ind[:,1],:]-self.wb_state[i]
            except:
                pass
    # def get_surrounding_terrain(self):
    #     wb_state=self.root_states_vec[:,0,:3]+self.env_origin

    #     for i, x in enumerate(wb_state):
    #         x_ind_prev=0
    #         y_ind_prev=0
    #         for j, xpts in enumerate(self.wb_xpts):
    #             for jj, ypts in enumerate(self.wb_ypts):
    #                 if jj==0:
    #                     x_ind=torch.searchsorted(self.tt[:,0,0], x[0]+xpts).item()
    #                     y_ind=torch.searchsorted(self.tt[0,:,1], x[1]+ypts).item()
    #                     x_ind_prev=copy.copy(x_ind)
    #                     y_ind_prev=copy.copy(y_ind)  
    #                 else:
    #                     x_ind=torch.searchsorted(self.tt[x_ind_prev-4:x_ind_prev+4,0,0], x[0]+xpts).item()
    #                     y_ind=torch.searchsorted(self.tt[0,y_ind_prev-4:y_ind_prev+4,1], x[1]+ypts).item()
    #                     x_ind_prev=copy.copy(x_ind)
    #                     y_ind_prev=copy.copy(y_ind)                                               
                    # self.scan_dot_buf[i,j,jj]=interp(x,self.tt[x_ind-1:x_ind+1,y_ind-1:y_ind+1,:])
    
    # def show_dots(self):
    #     box_pose = gymapi.Transform()
    #     box_size = 0.045
    #     asset_options = gymapi.AssetOptions()
    #     box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)        
    #     for i, env in enumerate(self.envs):
    #         box_pose.p.x = np.random.uniform(-0.2, 0.1)
    #         box_pose.p.y = np.random.uniform(-0.3, 0.3)
    #         box_pose.p.z = 1.5 * box_size
    #         box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    #         box_handle = self.gym.create_actor(env, box_asset, box_pose, "box", i, 0)   
    #         color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    #         self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)     
#             while
# def interp(dpts,pts):
#     # x1=pts[0,0,0]
#     # x2=pts[1,0,0]
#     # y1=pts[0,0,1]
#     # y2=pts[0,1,1]
#     int_mat=1/((pts[1,0,0]-pts[0,0,0])*(pts[0,1,1]-pts[0,0,1]))*torch.tensor([[pts[1,0,0]*pts[0,1,1],-pts[1,0,0]*pts[0,0,1],-pts[0,0,0]*pts[0,1,1],pts[0,0,0]*pts[0,0,1]],[-pts[0,1,1],pts[0,0,1],pts[0,1,1],-pts[0,0,1]],[-pts[1,0,0],pts[1,0,0],pts[0,0,0],-pts[0,0,0]],[1,-1,-1,1]],device="cuda:0")
#     a=int_mat@torch.tensor([pts[0,0,2],pts[0,1,2],pts[1,0,2],pts[1,1,2]],device="cuda:0")
#     return a[0]+a[1]*dpts[0]+a[2]*dpts[1]+a[3]*dpts[0]*dpts[1]
# def bisection(xval,mesh):
    
# rec_rew=[]
# if __name__ == '__main__':
#     tracked_dofs=[4]
#     tracked_root=[0,1,7]
    
#     envs=env(tracked_dofs=tracked_dofs,tracked_root=tracked_root)
#     envs._setup_env()
    
#     for i in range(100):
#         next_obs, reward, done, info=envs.step()
#         rec_rew.append(reward)
    # envs.reset()
    