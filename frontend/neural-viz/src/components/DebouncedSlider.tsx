import { useCallback, useEffect, useRef, useState } from 'react';

interface DebouncedSliderProps {
  initialValue: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  debounceMs?: number;
  onDebouncedChange: (value: number) => void;
  className?: string;
  style?: React.CSSProperties;
}

export const DebouncedSlider = ({
  initialValue,
  min = 0,
  max = 100,
  disabled = false,
  debounceMs = 300,
  onDebouncedChange,
  className,
  style,
}: DebouncedSliderProps) => {
  const [value, setValue] = useState(initialValue);
  const debounceTimeoutRef = useRef<number | null>(null);

  const handleSliderChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = Number(event.target.value);
    
    // Update internal state immediately for UI responsiveness
    setValue(newValue);
    
    // Clear existing timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    
    // Set up debounced call
    debounceTimeoutRef.current = window.setTimeout(() => {
      onDebouncedChange(newValue);
    }, debounceMs);
  }, [onDebouncedChange, debounceMs]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  return (
    <input
      type="range"
      min={min}
      max={max}
      value={value}
      onChange={handleSliderChange}
      disabled={disabled}
      className={className}
      style={style}
    />
  );
};