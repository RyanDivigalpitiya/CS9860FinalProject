# Detecting Human Presence in a Smart Home

### Recognizing Room Occupancy by using Convolutional Neural Networks with Inexpensive Thermal Sensors

> #### **Note:**
>
> In order to learn and challenge myself, I coded this project from scratch as much as I could, rather than use existing libraries. As a result, it may be hard to follow along. I plan on re-factoring this repo so that the codebase follows industry-standard practices, just like I did with my codebase in the CS9637 project repo.

Smart home technology has experienced rapid growth in the past two decades thanks to advancements in off-the-shelf products featuring Internet-of-Things capabilities and artificial intelligence-based functionalities. 
A core requirement for delivering basic features in a smart home usually revolves around detecting the presence of residents, either on a room-by-room basis or on a broader scale, such as whether people are home or away from home. Typical off-the-shelf smart home products that can bring such functionality to a home usually utilize motion sensors to detect presence. However, the use of motion sensors for the purpose of detecting if a room is occupied often fails when the person occupying the room is stationary. Examples include sitting on a sofa and watching tv, or simply sleeping at night in their bedroom.

In my project, I proposed making use of a person's heat signature to detect whether a room is occupied, by using cheap thermal sensors paired with a Convolutional Neural Network trained to differentiate between a person and false-positives signals found in a room, such as a hot-air vent.

------

### Professor's Feedback:

I received a grade of 100%. Feedback from my professor are copied & pasted below:

> *This project proposes a very interesting problem to detect human presence from a thermal sensor.
> The report is very well organized.
> The results are very promising with proper discussion. Probably because the data collected is less varied, the deep learning model only needs to learn certain fixed patterns to distinguish. Anyhow, it is a very novel project and well established. Hope you have more further study on this.*

