interface PatternSelectorProps {
    label: string;
    options: string[];
    value: string;
    onChange: (value: string) => void;
    autoFocus?: boolean;
    emptyMessage?: string;
    placeholder?: string;
}

export default function PatternSelector({
    label,
    options,
    value,
    onChange,
    autoFocus = false,
    emptyMessage = "No options added yet",
    placeholder = "Select an option..."
}: PatternSelectorProps) {
    if (options.length === 0) {
        return (
            <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '8px' }}>
                    {label}
                </label>
                <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '14px', fontStyle: 'italic', padding: '8px 0' }}>
                    {emptyMessage}
                </div>
            </div>
        );
    }

    return (
        <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', color: 'rgba(255,255,255,0.6)', fontSize: '12px', marginBottom: '8px' }}>
                {label}
            </label>
            
            {options.length < 5 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {options.map((option, index) => (
                        <label 
                            key={option} 
                            style={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                gap: '8px', 
                                color: '#fff',
                                fontSize: '14px',
                                cursor: 'pointer',
                                padding: '4px 0'
                            }}
                        >
                            <input
                                type="radio"
                                name={label.replace(/\s+/g, '_').toLowerCase()}
                                value={option}
                                checked={value === option}
                                onChange={(e) => onChange(e.target.value)}
                                style={{
                                    margin: 0,
                                    accentColor: '#3b82f6'
                                }}
                                autoFocus={autoFocus && index === 0}
                            />
                            {option}
                        </label>
                    ))}
                </div>
            ) : (
                <select 
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    style={{
                        width: '100%',
                        boxSizing: 'border-box',
                        background: '#0d0d14',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '6px',
                        padding: '8px 12px',
                        color: '#fff',
                        fontSize: '14px',
                        outline: 'none',
                    }}
                    autoFocus={autoFocus}
                >
                    <option value="">{placeholder}</option>
                    {options.map(option => (
                        <option key={option} value={option}>
                            {option}
                        </option>
                    ))}
                </select>
            )}
        </div>
    );
}