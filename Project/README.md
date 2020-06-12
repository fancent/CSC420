# NBA Player Detection

As me and my partner realised that there aren't any products for tracking individual basketball players that are available to the public, we decided to build our own and extract information from our favourite player, Luka Doncic.

We developed our own data set using [FFMPEG](https://ffmpeg.org/) and [LabelImg](https://github.com/tzutalin/labelImg).

For our model, we used Pytorch's Faster-RCNN as the base.

We also did basic MNIST detection using patch matching technique to extract the score.

After training on Google's Colaboratory, we were able to achieve detection with 99% accuracy with unseen video.

An example of our detection is as follow:

<img src="https://github.com/fancent/CSC420/blob/master/Project/predicted-1.gif" width="50%">

For full presentation and extra demo, please checkout the following YouTube videos:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=yAAhk3tyGTg
" target="_blank"><img src="http://img.youtube.com/vi/yAAhk3tyGTg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="49%" border="5" /> </a><a href="http://www.youtube.com/watch?feature=player_embedded&v=Vci6_vtE7gI" target="_blank"><img src="http://img.youtube.com/vi/Vci6_vtE7gI/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="49%" border="5" /></a>
