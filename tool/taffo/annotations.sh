extract_scalar_part() {
  local annotation="$1"

  if [[ "$annotation" == *"scalar("* ]]; then
    # extract everything starting from "scalar("
    local remainder=${annotation#*"scalar("}
    # initialize a counter for parentheses
    local count=1
    local scalar_part="scalar("
    # loop through each character after "scalar("
    for ((i=0; i<${#remainder}; i++)); do
      local char=${remainder:$i:1}
      scalar_part+="$char"
      # keep count of parentheses
      if [[ "$char" == "(" ]]; then
        ((count++))
      elif [[ "$char" == ")" ]]; then
        ((count--))
      fi
      # when count reaches zero, parentheses are balanced
      if [[ $count -eq 0 ]]; then
        break
      fi
    done

    echo "$scalar_part"
  else
    echo ""
  fi
}

extract_struct_part() {
  local annotation="$1"

  if [[ "$annotation" == *"struct["* ]]; then
    # extract everything starting from "struct["
    local remainder=${annotation#*"struct["}
    # initialize a counter for parentheses
    local count=1
    local struct_part="struct["
    # loop through each character after "struct["
    for ((i=0; i<${#remainder}; i++)); do
      local char=${remainder:$i:1}
      struct_part+="$char"
      # keep count of parentheses
      if [[ "$char" == "[" ]]; then
        ((count++))
      elif [[ "$char" == "]" ]]; then
        ((count--))
      fi
      # when count reaches zero, parentheses are balanced
      if [[ $count -eq 0 ]]; then
        break
      fi
    done

    echo "$struct_part"
  else
    echo ""
  fi
}
