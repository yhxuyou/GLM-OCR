#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "glmocr/common.h"
#include "glmocr/config.h"

namespace glmocr {

class ResultFormatter {
public:
    ResultFormatter(const ResultFormatterConfig& config);
    ~ResultFormatter() = default;
    
    // Format single OCR-only result (no layout)
    std::pair<std::string, std::string> format_ocr_result(
        const std::string& content,
        int page_idx = 0);
    
    // Format a complete result with region data from layout mode
    struct FormattedResult {
        std::string json_output;
        std::string markdown_output;
        std::map<std::string, std::shared_ptr<void>> image_files;
    };
    
    FormattedResult process(
        const std::vector<std::vector<Region>>& grouped_results,
        const std::map<std::tuple<int, int, int, int>, Image>& cropped_images = {});
    
private:
    ResultFormatterConfig config_;
    
    // Clean content (remove trailing whitespace, etc.)
    std::string clean_content(const std::string& content);
    
    // Map native label to standard type
    std::string map_label(const std::string& native_label);
    
    // Format region content based on type
    std::string format_region_content(
        const std::string& content,
        const std::string& label,
        const std::string& native_label);
    
    // Merge formula numbers with formulas
    std::vector<Region> merge_formula_numbers(std::vector<Region> regions);
    
    // Merge hyphenated text blocks
    std::vector<Region> merge_text_blocks(std::vector<Region> regions);
    
    // Format bullet points
    std::vector<Region> format_bullet_points(std::vector<Region> regions);
};

} // namespace glmocr
