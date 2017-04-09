/*----------------------------------------------------------------------------*/
/*  FICHERO:       calculaNormales.h									          */
/*  AUTOR:         Jorge Azorin											  */
/*													                          */
/*  RESUMEN												                      */
/*  ~~~~~~~												                      */
/* Fichero de definiciones y estructuras                                      */
/*    						                                                  */
/*----------------------------------------------------------------------------*/

#ifndef _NORMALES_H_
#define _NORMALES_H_

/*============================================================================ */
/* Constantes											                       */
/*============================================================================ */
#define ERRORCALC 1
#define OKCALC    0


/*============================================================================ */
/* Estructuras											                       */
/*============================================================================ */

struct sTPoint3D
{
	double x;
	double y;
	double z;
};
typedef struct sTPoint3D TPoint3D;


struct sTSurf
{
	int UPoints;
	int VPoints;
	TPoint3D** Buffer;
};
typedef struct sTSurf TSurf;


/*============================================================================ */
/* Variables Globales										                   */
/*============================================================================ */
TSurf S;
float* NormalUCPU;
float* NormalVCPU;
float* NormalWCPU;
float* NormalUGPU;
float* NormalVGPU;
float* NormalWGPU;



/*============================================================================ */
/* Funciones de tratamiento de memoria							 */
/*============================================================================ */
void BorrarSuperficie(void)
{
	int i;
	if (S.Buffer != NULL)
	{
		for (i = 0; i < S.VPoints; i++)
			if (S.Buffer[i] != NULL) free(S.Buffer[i]);
		free(S.Buffer);
		S.Buffer = NULL;
	}
}

int CrearSuperficie(int uPoints, int vPoints)
{
	int j;
	S.UPoints = uPoints;
	S.VPoints = vPoints;
	S.Buffer = (TPoint3D**)malloc(S.VPoints*sizeof(void*));
	if (S.Buffer == NULL) return ERRORCALC;
	for (j = 0; j < S.VPoints; j++)
	{
		S.Buffer[j] = (TPoint3D*)malloc(S.UPoints*(int)sizeof(TPoint3D));
		if (S.Buffer[j] == NULL)
		{
			BorrarSuperficie();
			return ERRORCALC;
		}
	}
	return OKCALC;
}


#endif // _NORMALES_H_

