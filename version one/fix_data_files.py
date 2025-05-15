#!/usr/bin/env python3
"""
Data File Validator and Fixer
This script validates and fixes data files for the Airspace Navigation Tool.
"""

import os
import sys

def validate_and_fix_nav_file(filename):
    """
    Validates and fixes a navigation points file.
    
    Args:
        filename (str): Path to the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Check for required header format
        has_header = False
        has_separator = False
        
        for i, line in enumerate(lines):
            if i == 0 and filename in line:
                has_header = True
            if line.strip() == "......":
                has_separator = True
                break
        
        if not has_header or not has_separator:
            # Fix the file
            fixed_lines = []
            fixed_lines.append(f"{os.path.basename(filename)}\n")
            fixed_lines.append("Navigation points\n")
            fixed_lines.append("......\n")
            
            # Add data lines
            for line in lines:
                if not (line.strip() == "......" or os.path.basename(filename) in line or "Navigation points" in line):
                    # Skip header-like lines and try to parse as navigation point
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            # Check if first part can be an integer
                            int(parts[0])
                            fixed_lines.append(line)
                        except ValueError:
                            # Skip lines that don't start with an ID
                            pass
            
            # Add trailing separator
            fixed_lines.append("......")
            
            # Write the fixed file
            with open(filename, 'w') as file:
                file.writelines(fixed_lines)
            
            print(f"Fixed {filename} format")
            return True
            
        return True
    except Exception as e:
        print(f"Error validating {filename}: {e}")
        return False

def validate_and_fix_seg_file(filename):
    """
    Validates and fixes a segments file.
    
    Args:
        filename (str): Path to the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Check for required header format
        has_header = False
        has_separator = False
        
        for i, line in enumerate(lines):
            if i == 0 and filename in line:
                has_header = True
            if line.strip() == "......":
                has_separator = True
                break
        
        if not has_header or not has_separator:
            # Fix the file
            fixed_lines = []
            fixed_lines.append(f"{os.path.basename(filename)}\n")
            fixed_lines.append("Segments that connect navigation points\n")
            fixed_lines.append("......\n")
            
            # Add data lines
            for line in lines:
                if not (line.strip() == "......" or os.path.basename(filename) in line or "Segments that connect" in line):
                    # Skip header-like lines and try to parse as segment
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            # Check if first two parts can be integers and third a float
                            int(parts[0])
                            int(parts[1])
                            float(parts[2])
                            fixed_lines.append(line)
                        except ValueError:
                            # Skip lines that don't match the expected format
                            pass
            
            # Add trailing separator
            fixed_lines.append("......")
            
            # Write the fixed file
            with open(filename, 'w') as file:
                file.writelines(fixed_lines)
            
            print(f"Fixed {filename} format")
            return True
            
        return True
    except Exception as e:
        print(f"Error validating {filename}: {e}")
        return False

def validate_and_fix_aer_file(filename):
    """
    Validates and fixes an airports file.
    
    Args:
        filename (str): Path to the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Check for required header format
        has_header = False
        has_separator = False
        
        for i, line in enumerate(lines):
            if i == 0 and filename in line:
                has_header = True
            if line.strip() == "......":
                has_separator = True
                break
        
        if not has_header or not has_separator:
            # Fix the file
            fixed_lines = []
            fixed_lines.append(f"{os.path.basename(filename)}\n")
            fixed_lines.append("Airports (in bold) with their SIDs and STARs list\n")
            fixed_lines.append("......\n")
            
            # Add data lines
            current_line = 0
            while current_line < len(lines):
                line = lines[current_line].strip()
                
                # Skip header-like lines
                if not (line == "......" or os.path.basename(filename) in line or "Airports" in line):
                    # Check if it's an airport code (4 uppercase letters)
                    parts = line.split()
                    if parts and len(parts[0]) == 4 and parts[0].isupper():
                        fixed_lines.append(f"{parts[0]}\n")
                        
                        # Check the next lines for SIDs and STARs
                        next_line = current_line + 1
                        while next_line < len(lines) and not (
                                lines[next_line].strip() and 
                                lines[next_line].strip().split() and 
                                len(lines[next_line].strip().split()[0]) == 4 and 
                                lines[next_line].strip().split()[0].isupper()):
                            sid_star_line = lines[next_line].strip()
                            
                            if sid_star_line and not sid_star_line.startswith('....'):
                                fixed_lines.append(f"{sid_star_line}\n")
                            
                            next_line += 1
                            if next_line >= len(lines):
                                break
                        
                        current_line = next_line - 1
                
                current_line += 1
            
            # Add trailing separator
            fixed_lines.append("......")
            
            # Write the fixed file
            with open(filename, 'w') as file:
                file.writelines(fixed_lines)
            
            print(f"Fixed {filename} format")
            return True
            
        return True
    except Exception as e:
        print(f"Error validating {filename}: {e}")
        return False

def main():
    """
    Main function to validate and fix all data files.
    """
    # Get the prefix from command line or use default
    prefix = "Cat"
    if len(sys.argv) > 1:
        prefix = sys.argv[1]
    
    # Define file paths
    nav_file = f"{prefix}_nav.txt"
    seg_file = f"{prefix}_seg.txt"
    aer_file = f"{prefix}_aer.txt"
    
    # Check if files exist
    files_exist = os.path.exists(nav_file) and os.path.exists(seg_file) and os.path.exists(aer_file)
    
    if not files_exist:
        print(f"Error: One or more files with prefix '{prefix}' not found.")
        return
    
    # Validate and fix files
    print(f"Validating and fixing files with prefix '{prefix}'...")
    
    nav_ok = validate_and_fix_nav_file(nav_file)
    seg_ok = validate_and_fix_seg_file(seg_file)
    aer_ok = validate_and_fix_aer_file(aer_file)
    
    if nav_ok and seg_ok and aer_ok:
        print(f"All files with prefix '{prefix}' have been validated and fixed if needed.")
    else:
        print(f"There were issues with one or more files.")

if __name__ == "__main__":
    main()