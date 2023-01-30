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

    - Create training script
    {
        - Case generator, without the OE for now.
        - test CNN [MEDIUM]
        - test GNN [MEDIUM]
        - test MLP [MEDIUM]
        - Parallelize the envs [HARD]
        {
            - With Everything will have LxBx...xN where L is the envs number and B is the B size
        }
    }
    - Add the padding to the FOV numpy array [EASY][DONE]
    - Find a way to implement obstacles into the mix [EASY]
    - For visuals, add the goal marker and the obstacles [EASY]
