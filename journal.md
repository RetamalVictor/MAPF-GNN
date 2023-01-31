# Log of the proyect

## 29/01

Done until now:

    - CBS implementation
    - Env with dynamic graph
    - Dataset Generator, but only the parser 

Today you finished, the cbs implementation to get the proper trayectories. They are parsed to numpy arrays and they generate the U*.

The env now has a starting position method.

For tomorrow, Prepare the training script, where every agent will move depending on the trayectory, but the model needs to predict the proper action.
I'll make a todo list for you:

    - Create training script[DONE]
    {
        - Case generator, without the OE for now.[DONE]
        - test CNN [MEDIUM][DONE]
        - test GNN [MEDIUM]
        - test MLP [MEDIUM][DONE]
        - Parallelize the envs [HARD][KINDA]
        {
            - With Everything will have LxBx...xN where L is the envs number and B is the B size
        }
    }
    - Add the padding to the FOV numpy array [EASY][DONE]
    - Find a way to implement obstacles into the mix [EASY]
    - For visuals, add the goal marker and the obstacles [EASY][DONE]

## 31/01

Automatic data generation ready
Also, the pipeline for cbs -> trayectory -> recordings is ready
Now is time for the training script
The data loader is ready!!
Create the train script

The thing is actually training with the CNN and MLP.

For tomorrow {
    - Add obstacles in the ENV
    - Create The accuraccy measurements
    -
}
