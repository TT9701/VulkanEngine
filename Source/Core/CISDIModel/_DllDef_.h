#pragma once

#ifdef CISDI_MODEL_DATA_EXPORTS
#define CISDI_MODEL_DATA_API __declspec(dllexport)
#else
#define CISDI_MODEL_DATA_API __declspec(dllimport)
#endif