import React, { useState } from 'react';

interface CategoryFilterProps {
  allCategories: string[];
  selectedCategories: string[];
  onCategoriesChange: (categories: string[]) => void;
}

export const CategoryFilter: React.FC<CategoryFilterProps> = ({
  allCategories,
  selectedCategories,
  onCategoriesChange,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleCategory = (category: string) => {
    if (selectedCategories.includes(category)) {
      onCategoriesChange(selectedCategories.filter(c => c !== category));
    } else {
      onCategoriesChange([...selectedCategories, category]);
    }
  };

  const isAllSelected = selectedCategories.length === 0;
  const displayText = isAllSelected 
    ? 'All Categories' 
    : selectedCategories.length === 1 
      ? `Category ${selectedCategories[0]}` 
      : `${selectedCategories.length} Categories`;

  return (
    <div style={{ position: 'relative' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '7px 10px',
        background: 'rgba(13, 13, 20, 0.88)',
        border: '1px solid rgba(255,255,255,0.09)',
        borderRadius: 10,
        backdropFilter: 'blur(10px)',
        boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
        cursor: 'pointer',
        minWidth: 120,
      }}
      onClick={() => setIsOpen(!isOpen)}>
        <span style={{
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'rgba(255,255,255,0.35)',
          marginRight: 2,
        }}>
          Filter
        </span>
        
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flex: 1,
          gap: 4,
        }}>
          <span style={{
            fontSize: 10,
            fontWeight: 400,
            color: 'rgba(255,255,255,0.7)',
            lineHeight: 1,
          }}>
            {displayText}
          </span>
          <span style={{
            fontSize: 8,
            color: 'rgba(255,255,255,0.4)',
            transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.15s ease',
          }}>
            ▼
          </span>
        </div>
      </div>

      {isOpen && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          right: 0,
          marginTop: 4,
          background: 'rgba(13, 13, 20, 0.95)',
          border: '1px solid rgba(255,255,255,0.09)',
          borderRadius: 8,
          backdropFilter: 'blur(10px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          zIndex: 1000,
          maxHeight: 200,
          overflowY: 'auto',
        }}>
          <button
            onClick={() => onCategoriesChange([])}
            style={{
              width: '100%',
              padding: '8px 12px',
              border: 'none',
              background: isAllSelected ? 'rgba(255,255,255,0.12)' : 'transparent',
              color: isAllSelected ? '#fff' : 'rgba(255,255,255,0.7)',
              cursor: 'pointer',
              fontSize: 10,
              fontWeight: isAllSelected ? 600 : 400,
              textAlign: 'left',
              borderRadius: 4,
              margin: 2,
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              if (!isAllSelected) {
                e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
              }
            }}
            onMouseLeave={(e) => {
              if (!isAllSelected) {
                e.currentTarget.style.background = 'transparent';
              }
            }}
          >
            All Categories
          </button>
          
          {allCategories.map((category) => {
            const isSelected = selectedCategories.includes(category);
            return (
              <button
                key={category}
                onClick={() => toggleCategory(category)}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: 'none',
                  background: isSelected ? 'rgba(255,255,255,0.12)' : 'transparent',
                  color: isSelected ? '#fff' : 'rgba(255,255,255,0.7)',
                  cursor: 'pointer',
                  fontSize: 10,
                  fontWeight: isSelected ? 600 : 400,
                  textAlign: 'left',
                  borderRadius: 4,
                  margin: 2,
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.background = 'transparent';
                  }
                }}
              >
                Category {category}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};