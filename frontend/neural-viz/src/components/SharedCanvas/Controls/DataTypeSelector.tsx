import { type FetcherType } from '../../../fetchers';
import { useFetcherType } from '../../../hooks/useFetcherType';

export const DataTypeSelector = () => {
  const { fetcherType, setFetcherType } = useFetcherType();

  return (
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
    }}>
      <span style={{
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: 'rgba(255,255,255,0.35)',
        marginRight: 2,
      }}>
        Data
      </span>

      {(['activation', 'saliency_map'] as FetcherType[]).map((type) => {
        const isActive = type === fetcherType;
        return (
          <button
            key={type}
            onClick={() => setFetcherType(type)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 9px',
              borderRadius: 6,
              border: isActive
                ? '1px solid rgba(255,255,255,0.3)'
                : '1px solid rgba(255,255,255,0.07)',
              background: isActive
                ? 'rgba(255,255,255,0.12)'
                : 'transparent',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <span style={{
              fontSize: 10,
              fontWeight: isActive ? 700 : 400,
              color: isActive ? '#fff' : 'rgba(255,255,255,0.4)',
              lineHeight: 1,
            }}>
              {type === 'activation' ? 'Activations' : 'Saliency Maps'}
            </span>
          </button>
        );
      })}
    </div>
  );
};
