import { useRef, useEffect } from 'react';

interface PatternAddFormProps {
    label: string;
    value: string;
    onChange: (value: string) => void;
    onSubmit: (e: React.FormEvent) => void;
    disabled?: boolean;
    placeholder?: string;
    buttonColor?: string;
    autoFocus?: boolean;
}

export default function PatternAddForm({
    label,
    value,
    onChange,
    onSubmit,
    disabled = false,
    placeholder = "Enter new pattern...",
    buttonColor = '#10b981',
    autoFocus = false
}: PatternAddFormProps) {
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (autoFocus && inputRef.current) {
            inputRef.current.focus();
        }
    }, [autoFocus]);

    return (
        <form onSubmit={onSubmit} style={{ marginBottom: '24px' }}>
            <label style={{ 
                display: 'block', 
                color: 'rgba(255,255,255,0.6)', 
                fontSize: '12px', 
                marginBottom: '4px' 
            }}>
                {label}
            </label>
            <div style={{ display: 'flex', gap: '8px' }}>
                <input 
                    ref={inputRef}
                    type="text" 
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    style={{
                        flex: 1,
                        background: '#0d0d14',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '6px',
                        padding: '8px 12px',
                        color: '#fff',
                        fontSize: '14px',
                        outline: 'none',
                    }}
                    placeholder={placeholder}
                    disabled={disabled}
                />
                <button 
                    type="submit"
                    disabled={disabled || !value.trim()}
                    style={{
                        padding: '8px 16px',
                        background: buttonColor,
                        border: 'none',
                        borderRadius: '6px',
                        color: '#fff',
                        cursor: 'pointer',
                        fontSize: '13px',
                        fontWeight: 600,
                        opacity: (disabled || !value.trim()) ? 0.5 : 1,
                    }}
                >
                    Add
                </button>
            </div>
        </form>
    );
}