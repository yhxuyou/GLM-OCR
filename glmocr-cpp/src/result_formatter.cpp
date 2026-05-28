#include "glmocr/result_formatter.h"
#include "glmocr/utils/string_utils.h"
#include <algorithm>
#include <cctype>

namespace glmocr {

using json = nlohmann::json;

ResultFormatter::ResultFormatter(const ResultFormatterConfig& config)
    : config_(config) {
}

std::string ResultFormatter::clean_content(const std::string& content) {
    if (content.empty()) return content;
    
    std::string result = content;
    
    // Remove leading/trailing whitespace
    result = utils::trim(result);
    
    // Clean repeated punctuation
    result = utils::clean_repeated_punctuation(result);
    
    // Normalize inline formulas
    result = utils::normalize_inline_formula(result);
    
    return result;
}

std::string ResultFormatter::map_label(const std::string& native_label) {
    for (const auto& [std_label, labels] : config_.label_visualization_mapping) {
        for (const auto& label : labels) {
            if (label == native_label) {
                return std_label;
            }
        }
    }
    return "text"; // default to text
}

std::string ResultFormatter::format_region_content(
    const std::string& content,
    const std::string& label,
    const std::string& native_label) {
    
    std::string result = clean_content(content);
    
    // Title formatting
    if (native_label == "doc_title") {
        // Remove leading #
        while (!result.empty() && result[0] == '#') {
            result = result.substr(1);
        }
        result = utils::trim(result);
        result = "# " + result;
    } else if (native_label == "paragraph_title") {
        // Remove leading - or #
        if (utils::starts_with(result, "- ")) {
            result = result.substr(2);
        }
        while (!result.empty() && result[0] == '#') {
            result = result.substr(1);
        }
        result = utils::trim(result);
        result = "## " + result;
    }
    
    // Formula formatting
    if (label == "formula") {
        // Remove leading/trailing formula delimiters
        if (utils::starts_with(result, "$$") && utils::ends_with(result, "$$")) {
            result = result.substr(2, result.size() - 4);
        } else if (utils::starts_with(result, "\\[") && utils::ends_with(result, "\\]")) {
            result = result.substr(2, result.size() - 4);
        } else if (utils::starts_with(result, "\\(") && utils::ends_with(result, "\\)")) {
            result = result.substr(2, result.size() - 4);
        }
        result = utils::trim(result);
        result = "$$\n" + result + "\n$$";
    }
    
    // Text formatting
    if (label == "text") {
        // Add closing code block if needed
        if (utils::starts_with(result, "```") && !utils::ends_with(result, "```")) {
            result += "\n```";
        }
        
        // Bullet point formatting
        if (utils::starts_with(result, "·") || 
            utils::starts_with(result, "•") || 
            utils::starts_with(result, "* ")) {
            result = "- " + result.substr(1);
            result = utils::trim(result);
        }
        
        // Format numbered lists (1., 2., etc.)
        if (result.size() > 2) {
            size_t dot_pos = result.find('.');
            if (dot_pos != std::string::npos && dot_pos > 0 && dot_pos < 5) {
                bool all_digits = true;
                for (size_t i = 0; i < dot_pos; ++i) {
                    if (!std::isdigit(result[i])) {
                        all_digits = false;
                        break;
                    }
                }
                if (all_digits) {
                    result = result.substr(0, dot_pos + 1) + " " + utils::trim(result.substr(dot_pos + 1));
                }
            }
        }
        
        // Format (1), (2), etc.
        if (result.size() > 3 && result[0] == '(') {
            size_t close_pos = result.find(')');
            if (close_pos != std::string::npos && close_pos > 1 && close_pos < 6) {
                bool valid = true;
                for (size_t i = 1; i < close_pos; ++i) {
                    if (!std::isdigit(result[i])) {
                        valid = false;
                        break;
                    }
                }
                if (valid) {
                    result = result.substr(0, close_pos + 1) + " " + utils::trim(result.substr(close_pos + 1));
                }
            }
        }
    }
    
    return result;
}

std::vector<Region> ResultFormatter::merge_formula_numbers(std::vector<Region> regions) {
    if (!config_.enable_merge_formula_numbers) return regions;
    
    std::vector<Region> result;
    std::vector<bool> skip(regions.size(), false);
    
    for (size_t i = 0; i < regions.size(); ++i) {
        if (skip[i]) continue;
        
        Region& region = regions[i];
        
        // Case 1: formula_number followed by formula
        if (region.native_label == "formula_number" && 
            i + 1 < regions.size() && 
            regions[i + 1].label == "formula") {
            
            Region merged = regions[i + 1];
            std::string num_content = utils::trim(region.content);
            // Remove parentheses
            if (utils::starts_with(num_content, "(") && utils::ends_with(num_content, ")")) {
                num_content = num_content.substr(1, num_content.size() - 2);
            } else if (utils::starts_with(num_content, "（") && utils::ends_with(num_content, "）")) {
                num_content = num_content.substr(1, num_content.size() - 2);
            }
            num_content = utils::trim(num_content);
            
            // Add tag to formula
            if (utils::ends_with(merged.content, "\n$$")) {
                merged.content = merged.content.substr(0, merged.content.size() - 3) + 
                                 " \\tag{" + num_content + "}\n$$";
            }
            
            result.push_back(merged);
            skip[i + 1] = true;
            continue;
        }
        
        // Case 2: formula followed by formula_number
        if (region.label == "formula" && 
            i + 1 < regions.size() && 
            regions[i + 1].native_label == "formula_number") {
            
            Region merged = region;
            std::string num_content = utils::trim(regions[i + 1].content);
            // Remove parentheses
            if (utils::starts_with(num_content, "(") && utils::ends_with(num_content, ")")) {
                num_content = num_content.substr(1, num_content.size() - 2);
            } else if (utils::starts_with(num_content, "（") && utils::ends_with(num_content, "）")) {
                num_content = num_content.substr(1, num_content.size() - 2);
            }
            num_content = utils::trim(num_content);
            
            // Add tag to formula
            if (utils::ends_with(merged.content, "\n$$")) {
                merged.content = merged.content.substr(0, merged.content.size() - 3) + 
                                 " \\tag{" + num_content + "}\n$$";
            }
            
            result.push_back(merged);
            skip[i + 1] = true;
            continue;
        }
        
        // Skip standalone formula numbers
        if (region.native_label == "formula_number") {
            continue;
        }
        
        result.push_back(region);
    }
    
    // Reindex
    for (size_t i = 0; i < result.size(); ++i) {
        result[i].index = static_cast<int>(i);
    }
    
    return result;
}

std::vector<Region> ResultFormatter::merge_text_blocks(std::vector<Region> regions) {
    if (!config_.enable_merge_text_blocks) return regions;
    
    std::vector<Region> result;
    std::vector<bool> skip(regions.size(), false);
    
    for (size_t i = 0; i < regions.size(); ++i) {
        if (skip[i]) continue;
        
        Region& region = regions[i];
        
        if (region.label != "text") {
            result.push_back(region);
            continue;
        }
        
        std::string content = region.content;
        bool merged = false;
        
        // Check if ends with hyphen
        std::string trimmed = utils::trim(content);
        if (!trimmed.empty() && trimmed.back() == '-') {
            // Look for next text block
            for (size_t j = i + 1; j < regions.size(); ++j) {
                if (regions[j].label == "text") {
                    std::string next_content = utils::trim(regions[j].content);
                    if (!next_content.empty() && std::islower(next_content[0])) {
                        // Check if it looks like a valid word merge
                        // (Simple heuristic - just merge)
                        
                        // Remove the hyphen
                        size_t hyphen_pos = content.find_last_of('-');
                        if (hyphen_pos != std::string::npos) {
                            content = content.substr(0, hyphen_pos);
                        }
                        
                        // Append next content
                        content += utils::trim(regions[j].content);
                        
                        Region merged_region = region;
                        merged_region.content = content;
                        result.push_back(merged_region);
                        skip[j] = true;
                        merged = true;
                        break;
                    }
                }
                break; // Only check immediate next block
            }
        }
        
        if (!merged) {
            result.push_back(region);
        }
    }
    
    // Reindex
    for (size_t i = 0; i < result.size(); ++i) {
        result[i].index = static_cast<int>(i);
    }
    
    return result;
}

std::vector<Region> ResultFormatter::format_bullet_points(std::vector<Region> regions) {
    if (!config_.enable_format_bullet_points) return regions;
    if (regions.size() < 3) return regions;
    
    for (size_t i = 1; i < regions.size() - 1; ++i) {
        Region& current = regions[i];
        
        // Only process text blocks
        if (current.native_label != "text") continue;
        
        // Check neighbors
        Region& prev = regions[i - 1];
        Region& next = regions[i + 1];
        
        if (prev.native_label != "text" || next.native_label != "text") continue;
        
        // Check if neighbors are bullet points
        bool prev_is_bullet = utils::starts_with(prev.content, "- ");
        bool next_is_bullet = utils::starts_with(next.content, "- ");
        
        if (prev_is_bullet && next_is_bullet) {
            // Check left alignment (using normalized bbox)
            const BBox& prev_bbox = prev.bbox;
            const BBox& curr_bbox = current.bbox;
            const BBox& next_bbox = next.bbox;
            
            // Simple check if x1 are close (within 10 normalized units)
            int threshold = 10;
            if (std::abs(curr_bbox.x1 - prev_bbox.x1) <= threshold &&
                std::abs(curr_bbox.x1 - next_bbox.x1) <= threshold) {
                
                // Add bullet point
                if (!utils::starts_with(current.content, "- ")) {
                    current.content = "- " + current.content;
                }
            }
        }
    }
    
    return regions;
}

std::pair<std::string, std::string> ResultFormatter::format_ocr_result(
    const std::string& content,
    int page_idx) {
    
    std::string cleaned = clean_content(content);
    
    // Build JSON
    json json_result = json::array({
        json::array({
            {{"index", 0}, {"label", "text"}, {"content", cleaned}, {"bbox_2d", nullptr}}
        })
    });
    
    return {json_result.dump(2, ' ', true), cleaned};
}

ResultFormatter::FormattedResult ResultFormatter::process(
    const std::vector<std::vector<Region>>& grouped_results,
    const std::map<std::tuple<int, int, int, int>, Image>& cropped_images) {
    
    FormattedResult result;
    json json_output = json::array();
    std::vector<std::string> markdown_pages;
    
    int image_counter = 0;
    
    for (size_t page_idx = 0; page_idx < grouped_results.size(); ++page_idx) {
        std::vector<Region> page_regions = grouped_results[page_idx];
        
        // Process each region
        std::vector<Region> processed_regions;
        for (const auto& region : page_regions) {
            Region processed = region;
            processed.native_label = region.native_label.empty() ? region.label : region.native_label;
            processed.label = map_label(processed.native_label);
            processed.content = format_region_content(
                region.content,
                processed.label,
                processed.native_label);
            
            // Check if it's an image region
            bool is_image_region = (processed.label == "image");
            processed.is_image = is_image_region;
            
            // Skip empty non-image regions
            if (!is_image_region && processed.content.empty()) {
                continue;
            }
            
            processed_regions.push_back(processed);
        }
        
        // Apply post-processing
        processed_regions = merge_formula_numbers(processed_regions);
        processed_regions = merge_text_blocks(processed_regions);
        processed_regions = format_bullet_points(processed_regions);
        
        // Reindex
        for (size_t i = 0; i < processed_regions.size(); ++i) {
            processed_regions[i].index = static_cast<int>(i);
        }
        
        // Build JSON for this page
        json page_json = json::array();
        std::vector<std::string> page_markdown;
        
        for (const auto& region : processed_regions) {
            json j;
            j["index"] = region.index;
            j["label"] = region.label;
            j["content"] = region.content;
            j["native_label"] = region.native_label;
            j["bbox_2d"] = json::array({region.bbox.x1, region.bbox.y1, region.bbox.x2, region.bbox.y2});
            
            // Handle images
            if (region.is_image) {
                auto key = std::make_tuple(
                    static_cast<int>(page_idx),
                    region.bbox.x1,
                    region.bbox.y1,
                    region.bbox.x2,
                    region.bbox.y2
                );
                
                // Note: key type mismatch, simplified for now
                // In real code, you'd look up the cropped image here
                
                std::string filename = "cropped_page" + std::to_string(page_idx) + 
                                       "_idx" + std::to_string(image_counter) + ".jpg";
                j["image_path"] = "imgs/" + filename;
                page_markdown.push_back("![Image " + std::to_string(page_idx) + "-" + 
                                        std::to_string(image_counter) + "](imgs/" + filename + ")");
                ++image_counter;
            } else {
                page_markdown.push_back(region.content);
            }
            
            page_json.push_back(j);
        }
        
        json_output.push_back(page_json);
        markdown_pages.push_back(utils::trim(utils::join(page_markdown, "\n\n")));
    }
    
    result.json_output = json_output.dump(2, ' ', true);
    result.markdown_output = utils::join(markdown_pages, "\n\n---\n\n");
    
    return result;
}

} // namespace glmocr
