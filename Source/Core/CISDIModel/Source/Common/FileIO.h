/**
 * @file FileIO.h
 * @author 
 * @brief 该文件定义了文件读写相关的帮助函数
 * @version 0.1
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <filesystem>

namespace IntelliDesign_NS::ModelData {

struct CISDI_3DModel;

/**
 * @brief 生成输出路径
 * @param input 源文件路径
 * @param output 输出文件路径
 * @return 输出文件的绝对路径
 */
::std::string ProcessOutputPath(const char* input, const char* output);

/**
 * @brief 将 CISDI_3DModel 写入至文件
 * @param outputPath 输出文件路径
 * @param data CISDI_3DModel::Header 对象
 */
void Write_CISDI_File(const char* outputPath, CISDI_3DModel const& data);

/**
 * @brief 读取文件至 CISDI_3DModel 对象
 * @param model 输出 CISDI_3DModel 对象指针
 * @param path 文件路径
 * @param pMemPool pmr memory resource 指针
 * @return CISDI_3DModel 对象
 */
void Read_CISDI_File(CISDI_3DModel* model, const char* path,
                     ::std::pmr::memory_resource* pMemPool);

}  // namespace IntelliDesign_NS::ModelData