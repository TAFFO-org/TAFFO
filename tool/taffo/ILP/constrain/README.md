## Constrain Grammar in EBFN




*start* → [*exp*__,__]\* *exp* | **ε**
*exp*   →  **N**_item_
*exp*   →  *stype*\__item_
*exp*   →  *mtype*\__item_\__item_
*stype* →  **ADD**|**SUB**|**MUL**|**DIV**|**REM**	
*mtype* →  **CAST**
*item*  →  **HALF**|**FIX**|**FLOAT**|**DOUBLE**|**QUAD**|**FP80**|**PPC128**|**BF16**
