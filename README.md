# Case Studies for Learn to Discover! :brain: > :world_map: > :robot: 

## :bookmark_tabs: Table of content
1. Deep learning 
	* Open [1-Intro_dl_MLPs.ipynb](deep_learning_mab/Improved_notebooks/1-Intro_dl_MLPs.ipynb) in GitHub. (✅  run in codespace)   
	* Open [Intro_dl_CNNs.ipynb](deep_learning_mab/MAB_New_Edits/Intro_dl_CNNs.ipynb) in GitHub. (✅ run in codespace)   
2. Reinforcement learning 
	* Open [session1-tabular.ipynb](reinforcement_learning_Neythen/updated-notebooks/session1-tabular.ipynb) in GitHub. (✅  run in codespace)
	* Open [session2-deep.ipynb](reinforcement_learning_Neythen/updated-notebooks/session2-deep.ipynb) in GitHub. (✅  run in codespace)
3. Classifier for two groups of proteins

## :star2: Getting started!

### :computer: In your local machine 
Create your conda environment as suggested [here](dependencies/README.md).   
Then you can launch your jupyter notebooks.
``` 
mamba activate l2dVE && jupyter notebook --browser=firefox
```

### :cloud: Using Github Codespaces
[Codespaces](https://docs.github.com/en/codespaces/overview) is a development environment hosted in the cloud. 
We have created a config file, [devcontainer.json](devcontainer/devcontainer.json) that currently sopports `2-core • 8GB RAM • 32GB` and added NVIDIA CUDA features, however the GPU option is only available for selected customers as trial period [:link:](https://docs.github.com/en/enterprise-cloud@latest/codespaces/developing-in-a-codespace/getting-started-with-github-codespaces-for-machine-learning).

1. Create codespace: Go to `Code` icon and select `create codespace on PREFERED BRANCH`. Taking 4-ish minutes to set up!  
	1.1 You might already have an image, in which case just activate `docker_funny_name_ID` at https://github.com/codespaces    
	1.2 The default codespace is configured with `2-core • 8GB RAM • 32GB` but you can choose other options (4, 8, 16 cores, etc)  
	1.3 Once the setup is complete, a Visual Studio Code IDE will show up on your browser     
2. In the `EXPLORER` panel to the left, open the relevant notebook     
3. In the notebook view, on the top right, click on `Select Kernel`  
	3.1 From the drop down menu in the top centre, select: `Install enable suggested extensions Python + Jupyter`    
	3.2 Select: `Python environments...` (you may have to click on `Select Kernel` again)    
	3.3 From the drop down menu, select: `l2dVE (Python VERSION) /opt/conda/envs/l2dVE/bin/python`    
4. Run cells in notebook and have fun! 🚀  
5. You might like to commit changes. For this we suggest creating specific branches to avoid conflicts with the `main` branch   
6. When you are done, just stop running the container. Click on the bottom left menu `Codespaces: funny_name_ID`, select `Stop Current Codebase`  
7. You might like to go to https://github.com/codespaces/ to delete your container `docker_funny_name_ID` by clicking in three dots and delete it  

Alternatively, you can open the notebook you would like to run on Github and click on the codespace badge (which looks like this ![badge icon](https://github.com/codespaces/badge.svg)) at the top. Then fill in the setup parameters on the browser:
- Select the branch you would like to use 
- Select the Dev container configuration (Miniconda Python 3)
- The defaults for the region and machine type should work fine for us!
Then continue with the setup from step 2 above.

## :octocat: Clone repository
Clone the repository by typing (or copying) the following line in a terminal at your selected path in your machine.
You might need to generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). 
```
git clone git@github.com:LearnToDiscover/Case_Studies_L2D.git
```

## :family: Contributors
Thanks goes to all these people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):  
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<!-- ADD GITHUB USERNAME AND HASH FOR GITHUB PHOTO -->
		<a href="https://github.com/???"><img src="https://avatars1.githubusercontent.com/u/23114020?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>ADD NAME SURNAME</b> </sub>        
		</a>
		<br />
			<!-- ADD GITHUB REPOSITORY AND PROJECT, TITLE AND EMOJIS -->
			<a href="https://github.com/$PROJECTNAME/$REPOSITORY_NAME/commits?author=" title="Research">  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<!-- ADD GITHUB USERNAME AND HASH FOR GITHUB PHOTO -->
		<a href="https://github.com/zcqsntr"><img src="https://avatars1.githubusercontent.com/u/33317183?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Neythen Treloar</b> </sub>        
		</a>
		<br />
			<!-- ADD GITHUB REPOSITORY AND PROJECT, TITLE AND EMOJIS -->
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=zcqsntr" title="Code"> </a> 
			<a href="ttps://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=zcqsntr" title="Research and Documentation"> </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<!-- ADD GITHUB USERNAME AND HASH FOR GITHUB PHOTO -->
		<a href="https://github.com/maalbadri"><img src="https://avatars1.githubusercontent.com/u/43252757?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Mohamed Ali Al-Badri</b> </sub>        
		</a>
		<br />
			<!-- ADD GITHUB REPOSITORY AND PROJECT, TITLE AND EMOJIS -->
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=maalbadri" title="Code"> </a> 
			<a href="ttps://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=maalbadri" title="Research and Documentation"> </a>
	</td>
        <!-- CONTRIBUTOR -->
	<td align="center">
		<!-- ADD GITHUB USERNAME AND HASH FOR GITHUB PHOTO -->
		<a href="https://github.com/Lgpoll"><img src="https://avatars1.githubusercontent.com/u/122795890?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Louise Pollock</b> </sub>        
		</a>
		<br />
			<!-- ADD GITHUB REPOSITORY AND PROJECT, TITLE AND EMOJIS -->
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=Lgpoll" title="Code"> </a> 
			<a href="ttps://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=Lgpoll" title="Research and Documentation"> </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/sanazjb"><img src="https://avatars1.githubusercontent.com/u/31011905?v=4?s=100" width="100px;" alt=""/>
			<br />
			<sub><b>Sanaz Jabbari</b></sub>          
			<br />
		</a>
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=sanazjb" title="Code"> </a> 
			<a href="ttps://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=sanazjb" title="Research and Documentation"> </a>
	</td>
        <!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/edlowther"><img src="https://avatars1.githubusercontent.com/u/7374954?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Ed Lowther</b> </sub>        
		</a>
		<br />
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=edlowther" title="Code"> </a> 
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=edlowther" title="Research and Documentation"> </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/sfmig"><img src="https://avatars1.githubusercontent.com/u/33267254?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Sofia Miñano</b> </sub>        
		</a>
		<br />
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=sfmig" title="Code"> </a> 
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=sfmig" title="Research and Documentation"> </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/mxochicale"><img src="https://avatars1.githubusercontent.com/u/11370681?v=4?s=100" width="100px;" alt=""/>
			<br />
			<sub><b>Miguel Xochicale</b></sub>          
			<br />
		</a>
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=mxochicale" title="Code"> </a> 
			<a href="ttps://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=mxochicale" title="Research and Documentation"> </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/dpshelio"><img src="https://avatars1.githubusercontent.com/u/963242?v=4?s=100" width="100px;" alt=""/>
			<br />
			<sub><b>David Pérez-Suárez</b></sub>          
			<br />
		</a>
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=dpshelio" title="Code"> </a> 
			<a href="ttps://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=dpshelio" title="Research and Documentation">  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<!-- ADD GITHUB USERNAME AND HASH FOR GITHUB PHOTO -->
		<a href="https://github.com/DrAdamLee"><img src="https://avatars1.githubusercontent.com/u/93711955?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Adam Lee</b> </sub>        
		</a>
		<br />
			<!-- ADD GITHUB REPOSITORY AND PROJECT, TITLE AND EMOJIS -->
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=DrAdamLee" title="Research">  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<!-- ADD GITHUB USERNAME AND HASH FOR GITHUB PHOTO -->
		<a href="https://github.com/sabaferdous12"><img src="https://avatars1.githubusercontent.com/u/7863996?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Saba Ferdous</b> </sub>        
		</a>
		<br />
			<!-- ADD GITHUB REPOSITORY AND PROJECT, TITLE AND EMOJIS -->
			<a href="https://github.com/LearnToDiscover/Case_Studies_L2D/commits?author=" title="Research">  </a>
	</td>
  </tr>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This work follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.  
Contributions of any kind welcome!
